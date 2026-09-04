#!/usr/bin/env bash
# Raw TCP bandwidth of the two DRANET NICs on ct6e-standard-4t, with neper.
#
# Establishes the transport ceiling that ppermute / psum are measured against:
# no JAX, no MegaScale, just tcp_stream between two pods that hold the same
# eth1 / eth2 the benchmark pods held.
#
# Method and flags follow google/dranet docs/user/gke-tpu-performance.md, whose
# published reference on this machine type is 180.17 + 174.73 = 354.9 Gbps.
#
# Three measurements per claim config:
#   eth1   one NIC alone
#   eth2   the other NIC alone
#   both   both NICs at once  <- the number that matters
#
# DIR selects directionality, and this matters for which collective the result
# is a fair ceiling for:
#   DIR=bidi  client and server both read+write (-rw). Peer for `psum`, which
#             also sends and receives concurrently. This is what the dranet doc
#             uses. `throughput`/`local_throughput` and `remote_throughput` are
#             the two directions.
#   DIR=uni   client writes only, server reads only. Peer for `ppermute_uni`,
#             where rank 0 only sends. NOTE: neper counts on the READ side, so
#             the writer logs local_throughput=0 and the real figure is in
#             remote_throughput.
set -uo pipefail

CLUSTER=dcnbw-mn-ew4
ZONE=europe-west4-a
PROJECT=tpu-launchpad-playground
LEN="${LEN:-60}"                 # seconds per test, dranet doc uses 60
THREADS="${THREADS:-16}"
FLOWS="${FLOWS:-32}"
CONFIGS="${CONFIGS:-2-netdev 2-netdev-tuned}"
DIR="${DIR:-bidi}"                # bidi | uni
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/results"
export KUBECONFIG="$HERE/../../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }
K()   { kubectl "$@"; }

mkdir -p "$OUT"
tok
K apply -f "$HERE/claim-tuned.yaml" >/dev/null

for CLAIM in $CONFIGS; do
  say "################ claim=$CLAIM  dir=$DIR"
  K delete pod neper-0 neper-1 --ignore-not-found --wait=true >/dev/null 2>&1
  sed "s|__CLAIM__|$CLAIM|g" "$HERE/neper-pods.yaml" | K apply -f - >/dev/null

  ready=0
  for i in $(seq 1 40); do
    r=$(K get pods -l app=neper --no-headers 2>/dev/null | awk '$3=="Running"' | wc -l)
    [ "$r" = "2" ] && { ready=1; break; }
    sleep 10
  done
  if [ "$ready" != "1" ]; then
    say "pods not Running:"; K get pods -l app=neper -o wide 2>&1 | sed 's/^/  /'
    K describe pod neper-0 2>&1 | sed -n '/^Events:/,$p' | head -12 | sed 's/^/  /'
    continue
  fi

  # what the pod actually got
  say "--- interfaces in neper-1 (server) ---"
  K exec neper-1 -- sh -c 'ip -br -4 addr show; for i in eth1 eth2; do echo -n "$i mtu="; cat /sys/class/net/$i/mtu; echo -n "$i speed="; cat /sys/class/net/$i/speed; done' 2>&1 | sed 's/^/  /'
  K exec neper-1 -- sh -c 'ip -br -4 addr show' > "$OUT/$DIR.$CLAIM.ifaces.txt" 2>&1

  S1=$(K exec neper-1 -- sh -c "ip -4 -o addr show eth1 | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null | tr -d '\r\n')
  S2=$(K exec neper-1 -- sh -c "ip -4 -o addr show eth2 | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null | tr -d '\r\n')
  say "server eth1=$S1  eth2=$S2"
  [ -z "$S1" ] || [ -z "$S2" ] && { say "could not read server IPs, skipping"; continue; }

  # $1 = label, remaining = list of "index:server_ip" to drive concurrently
  run_test() {
    local label="$1"; shift
    local pairs=("$@")
    local srvcmd="" clicmd=""
    for p in "${pairs[@]}"; do
      local i="${p%%:*}" ip="${p##*:}"
      local C=$((52279+i)) P=$((38339+i))
      local sflag cflag
      if [ "$DIR" = "uni" ]; then sflag="-r"; cflag="-w"; else sflag="-rw"; cflag="-rw"; fi
      srvcmd+="tcp_stream -C$C --port=$P --skip-rx-copy $sflag -Z -B16384 --test-length=$((LEN+15)) --suicide-length=$((LEN+60)) -F100 --num-threads=$THREADS --num-flows=$FLOWS -D0 --logtostderr &> /tmp/s$i.log & "
      clicmd+="tcp_stream -C$C --port=$P --skip-rx-copy $cflag -Z -B16384 --test-length=$LEN --suicide-length=$((LEN+30)) -F100 --num-threads=$THREADS --num-flows=$FLOWS --client -H $ip -D0 --logtostderr &> /tmp/c$i.log & "
    done
    say "  running $label ..."
    K exec neper-1 -- sh -c "pkill tcp_stream; rm -f /tmp/s*.log; $srvcmd sleep 2" >/dev/null 2>&1
    sleep 3
    K exec neper-0 -- sh -c "pkill tcp_stream; rm -f /tmp/c*.log; $clicmd wait" >/dev/null 2>&1
    local total=0
    for p in "${pairs[@]}"; do
      local i="${p%%:*}"
      K exec neper-0 -- cat /tmp/c$i.log > "$OUT/$DIR.$CLAIM.$label.c$i.log" 2>/dev/null
      local g
      # neper counts on the read side. In uni mode the client only writes, so
      # its local throughput is 0 and the real number is the server's read rate.
      g=$(grep -m1 '^throughput=' "$OUT/$DIR.$CLAIM.$label.c$i.log" 2>/dev/null | cut -d= -f2)
      if [ -z "$g" ] || [ "$g" = "0" ] || [ "$g" = "0.00" ]; then
        g=$(grep -m1 '^remote_throughput=' "$OUT/$DIR.$CLAIM.$label.c$i.log" 2>/dev/null | cut -d= -f2)
        g=$(awk -v b="${g:-0}" 'BEGIN{print b/1e6}')      # bit/s -> Mbit/s
      fi
      [ -z "$g" ] && g=0
      awk -v i="$i" -v g="$g" 'BEGIN{printf "      stream %s : %10.2f Mbit/s = %6.1f Gbps\n", i, g, g/1000}'
      total=$(awk -v t="$total" -v g="$g" 'BEGIN{print t+g}')
    done
    awk -v l="$label" -v t="$total" 'BEGIN{printf "    %-6s TOTAL : %6.1f Gbps\n", l, t/1000}'
    awk -v d="$DIR" -v c="$CLAIM" -v l="$label" -v t="$total" 'BEGIN{printf "%s %s %s %.1f\n", d, c, l, t/1000}' >> "$OUT/summary.txt"
    K exec neper-1 -- sh -c 'pkill tcp_stream' >/dev/null 2>&1
    sleep 3
  }

  : > /dev/null
  run_test eth1 "0:$S1"
  run_test eth2 "1:$S2"
  run_test both "0:$S1" "1:$S2"
done

tok
K delete pod neper-0 neper-1 --ignore-not-found >/dev/null 2>&1
say "================ SUMMARY (Gbps) ================"
column -t "$OUT/summary.txt" 2>/dev/null || cat "$OUT/summary.txt"
say "DONE"
