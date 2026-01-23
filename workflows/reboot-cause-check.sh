#!/usr/bin/env bash
# reboot-cause-check.sh
# 意図しない再起動の原因を特定するための証跡を収集し、要約を出力します。

set -euo pipefail

WINDOW_MIN=30   # 直前ブート時刻の前後に見る分数
OUT="/var/tmp/reboot-report-$(date +%Y%m%d%H%M).txt"

while getopts "w:h" opt; do
  case "$opt" in
    w) WINDOW_MIN="${OPTARG}" ;;
    h)
      echo "Usage: sudo bash $0 [-w WINDOW_MIN]"
      exit 0
      ;;
  esac
done

log() { echo -e "$@" | tee -a "$OUT" >/dev/null; }
hdr() { log "\n===== $1 ====="; }
sep() { log "\n--- $1 ---"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

# 1) 基本情報
: >"$OUT"
hdr "基本情報"
HOST="$(hostname -f 2>/dev/null || hostname)"
log "Host: $HOST"
log "Date: $(date -R)"
log "Uname: $(uname -a)"
if command_exists hostnamectl; then
  sep "hostnamectl"
  hostnamectl 2>&1 | tee -a "$OUT" >/dev/null
fi
log "Uptime: $(uptime -p || true)"

# 2) 直近の再起動・ブート時刻
hdr "直近の再起動・ブート履歴"
LAST_BOOT="$(who -b 2>/dev/null | awk '{print $3, $4}' || true)"
log "who -b => $LAST_BOOT"
sep "last -x (最新 10 件)"
last -x | head -n 10 | tee -a "$OUT" >/dev/null

# 直前ブートの開始時刻(≒前回シャットダウン直後)を推定
# journald ベースで取得できなければ who -b を使用
PREV_BOOT_JOURNAL_TS=""
if command_exists journalctl; then
  # 現在ブート(-b 0)の開始時刻
  CUR_BOOT_START="$(journalctl -b -0 --no-pager -n 1 2>/dev/null | head -n1 | sed -E 's/^([A-Z][a-z]{2} +[0-9 ]{1,2} [0-9:]{8}).*/\1/' || true)"
  # 直前ブート(-b -1)のタイムスタンプ範囲
  PREV_BOOT_FIRST="$(journalctl -b -1 --no-pager 2>/dev/null | head -n1 | sed -E 's/^([A-Z][a-z]{2} +[0-9 ]{1,2} [0-9:]{8}).*/\1/' || true)"
  PREV_BOOT_LAST="$(journalctl -b -1 --no-pager 2>/dev/null | tail -n1 | sed -E 's/^([A-Z][a-z]{2} +[0-9 ]{1,2} [0-9:]{8}).*/\1/' || true)"

  sep "journalctl ブート境界"
  log "前回ブート: $PREV_BOOT_FIRST 〜 $PREV_BOOT_LAST"
  log "今回ブート開始: $CUR_BOOT_START"

  # 直前ブートの“終了付近”を中心に前後 WINDOW_MIN 分を見る
  PREV_BOOT_JOURNAL_TS="$PREV_BOOT_LAST"
fi

CENTER_TS="${PREV_BOOT_JOURNAL_TS}"
if [[ -z "$CENTER_TS" && -n "$LAST_BOOT" ]]; then
  CENTER_TS="$LAST_BOOT"
fi

if [[ -z "$CENTER_TS" ]]; then
  log "※ 中心時刻の特定に失敗しました。デフォルトで過去 ${WINDOW_MIN} 分を走査します。"
  CENTER_TS="now"
fi

# 3) 重要ログ抽出（再起動前後 ±WINDOW_MIN 分）
hdr "再起動前後ログ（±${WINDOW_MIN}分）"
SINCE_OPT=""
UNTIL_OPT=""

# GNU date / BSD date 両対応をざっくり試みる
ts_to_since_until() {
  local center="$1"
  local since until
  if date -d "$center" >/dev/null 2>&1; then
    since="$(date -d "$center - ${WINDOW_MIN} min" '+%Y-%m-%d %H:%M:%S')"
    until="$(date -d "$center + ${WINDOW_MIN} min" '+%Y-%m-%d %H:%M:%S')"
  else
    since="$(date -j -v-"${WINDOW_MIN}"M -f "%b %e %T %Y" "$center $(date +%Y)" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "")"
    until="$(date -j -v+"${WINDOW_MIN}"M -f "%b %e %T %Y" "$center $(date +%Y)" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "")"
  fi
  echo "$since|$until"
}

IFS='|' read -r SINCE_OPT UNTIL_OPT < <(ts_to_since_until "$CENTER_TS" || true)
[[ -z "$SINCE_OPT" ]] && SINCE_OPT="$(date -d "now - ${WINDOW_MIN} min" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
[[ -z "$UNTIL_OPT" ]] && UNTIL_OPT="$(date -d "now + ${WINDOW_MIN} min" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"

log "対象期間: since=\"$SINCE_OPT\" until=\"$UNTIL_OPT\""

if command_exists journalctl; then
  sep "journalctl 抜粋（重要ワードフィルタ）"
  journalctl --since "$SINCE_OPT" --until "$UNTIL_OPT" --no-pager \
  | egrep -i "panic|oops|BUG:|watchdog|NMI|backtrace|stack trace|thermal|overheat|Machine check|MCE|mcelog|I/O error|EXT4-fs error|btrfs: error|xfs_(err|alert)|Out of memory|oom-killer|invoked oom-killer|power|ACPI.*Critical|PCIe Bus Error|resetting link|segfault|general protection fault|RCU stall|ktime|soft lockup|hard lockup|kexec|killed by SIGKILL|systemd-logind|systemctl reboot|shutdown" \
  | tee -a "$OUT" >/dev/null || true

  sep "journalctl -b -1 末尾50行（直前ブートの最後）"
  journalctl -b -1 --no-pager -n 50 2>/dev/null | tee -a "$OUT" >/dev/null || true
fi

if [[ -f /var/log/messages ]]; then
  sep "/var/log/messages 抜粋（重要ワードフィルタ）"
  awk -v since="$SINCE_OPT" -v until="$UNTIL_OPT" '1==1{print}' /var/log/messages \
  | egrep -i "panic|oops|BUG:|watchdog|NMI|I/O error|EXT4-fs error|xfs_(err|alert)|btrfs: error|Out of memory|oom-killer|thermal|ACPI.*Critical|power|segfault" \
  | tail -n 300 | tee -a "$OUT" >/dev/null || true
fi

# 4) OOM, Kernel panic, Filesystem, HW/温度、ユーザ操作などの兆候を個別要約
hdr "検出サマリ（ヒューリスティック）"
summary_flag=0

detect_and_report() {
  local name="$1" pattern="$2"
  local hit=""
  if command_exists journalctl; then
    hit="$(journalctl --since "$SINCE_OPT" --until "$UNTIL_OPT" --no-pager | egrep -i "$pattern" || true)"
  fi
  if [[ -z "$hit" && -f /var/log/messages ]]; then
    hit="$(egrep -i "$pattern" /var/log/messages || true)"
  fi
  if [[ -n "$hit" ]]; then
    log "[!] 兆候: $name"
    echo "$hit" | head -n 10 | sed 's/^/    /' | tee -a "$OUT" >/dev/null
    summary_flag=1
  else
    log "[-] $name: 兆候なし"
  fi
}

detect_and_report "Kernel panic/BUG/Oops" "panic|BUG:|Oops:"
detect_and_report "Watchdog/Lockup" "watchdog|soft lockup|hard lockup|RCU stall"
detect_and_report "OOM (メモリ逼迫)" "Out of memory|oom-killer|invoked oom-killer"
detect_and_report "ファイルシステム/ディスクI/O" "I/O error|EXT4-fs error|xfs_(err|alert)|btrfs: error|resetting link"
detect_and_report "温度/電源/ACPI" "thermal|overheat|ACPI.*Critical|power"
detect_and_report "ユーザ/プロセスによる再起動" "systemctl reboot|shutdown|reboot|systemd-logind.*Power|User requested reboot|Received SIGINT|halt"

# 5) クラッシュダンプ、MCE、BMC/IPMI
hdr "追加の低レベル情報"

if [[ -d /var/crash ]]; then
  sep "/var/crash 内容"
  ls -ltrh /var/crash | tee -a "$OUT" >/dev/null || true
else
  log "/var/crash: ディレクトリなし（kdump 未設定の可能性）"
fi

if command_exists mcelog; then
  sep "mcelog --ascii （直近50行）"
  mcelog --ascii 2>/dev/null | tail -n 50 | tee -a "$OUT" >/dev/null || true
fi

if command_exists ipmitool; then
  sep "ipmitool sel list （最新20件）"
  ipmitool sel list 2>/dev/null | tail -n 20 | tee -a "$OUT" >/dev/null || true
fi

# 6) 監査ログ（auditd）からの再起動操作痕跡
if command_exists ausearch; then
  hdr "auditd からの操作痕跡（ SYSTEM_SHUTDOWN / USER_SHUTDOWN ）"
  ausearch -m SYSTEM_SHUTDOWN,USER_SHUTDOWN -ts "$SINCE_OPT" -te "$UNTIL_OPT" 2>/dev/null \
    | tail -n 200 | tee -a "$OUT" >/dev/null || true
fi

# 7) dmesg 末尾
hdr "dmesg 末尾（最新 80 行）"
dmesg -T 2>/dev/null | tail -n 80 | tee -a "$OUT" >/dev/null || true

# 8) 結論テンプレート
hdr "暫定結論"
if [[ "$summary_flag" -eq 1 ]]; then
  log "上記サマリの『[!] 兆候』に該当する項目が再起動の直接原因/誘因の可能性があります。"
else
  log "再起動直前に顕著な異常ログは見つかりませんでした。"
  log "・電源/ハイパーバイザ側イベント（ESXi/Proxmox/KVM等）の確認"
  log "・BMC/IPMI SEL、UPSログ、ホスト管理基盤（Ansible/手動オペ）の履歴確認"
  log "・次回に向け kdump 有効化とクラッシュダンプ採取 を推奨します。"
fi

log "\n出力レポート: $OUT"
echo -e "\n[OK] 調査レポートを $OUT に出力しました。"
