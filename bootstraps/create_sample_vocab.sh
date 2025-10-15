#!/bin/bash

# サンプル語彙ファイルを作成するスクリプト
# 使用方法: ./create_sample_vocab.sh

VOCAB_DIR="assets/molecules"
VOCAB_FILE="$VOCAB_DIR/vocab.txt"

echo "サンプル語彙ファイルを作成中..."

# ディレクトリが存在しない場合は作成
mkdir -p "$VOCAB_DIR"

# 基本的なSMILES語彙を作成
cat > "$VOCAB_FILE" << 'EOF'
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
[unused0]
[unused1]
[unused2]
[unused3]
[unused4]
C
N
O
S
P
F
Cl
Br
I
c
n
o
s
p
1
2
3
4
5
6
7
8
9
(
)
[
]
=
#
+
-
@
@@
H
.
/
\
%
EOF

echo "✓ サンプル語彙ファイルを作成しました: $VOCAB_FILE"
echo "このファイルには基本的なSMILES記法のトークンが含まれています。"
echo "実際の使用では、トレーニングデータから生成された語彙ファイルを使用してください。"
