#!/bin/bash
# Download GovInfo BILLSTATUS bulk files into data/billstatus_bulk/<congress>/<type>/.
# Congresses 108–119 (the 119th is still in session; exclude it from training data).

cd "$(dirname "$0")" || exit 1
OUT=billstatus_bulk

for congress in {108..119}; do
  for type in hr s hres sres hjres sjres hconres sconres; do

    echo "Downloading Congress $congress, type $type..."

    mkdir -p "$OUT/$congress/$type"

    curl -fL -C - --retry 3 \
      "https://www.govinfo.gov/bulkdata/BILLSTATUS/$congress/$type/BILLSTATUS-$congress-$type.zip" \
      -o "$OUT/$congress/$type.zip"

    if [ $? -eq 0 ]; then
      unzip -q -o "$OUT/$congress/$type.zip" \
        -d "$OUT/$congress/$type"
    else
      echo "Failed: Congress $congress, type $type"
    fi

  done
done

echo "All downloads completed!"
