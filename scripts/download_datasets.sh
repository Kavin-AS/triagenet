#!/usr/bin/env bash
set -u

# TriageNet Phase 1.5 data acquisition gate.
#
# Automatable today:
# - CALCE CS2/CX2 LCO prismatic cells from https://calce.umd.edu/battery-data.
#   The page lists CS2 capacity as 1100 mAh and CX2 capacity as 1350 mAh, and links each
#   cell to zip archives under https://web.calce.umd.edu/batteries/data/<CELL>.zip.
#
# Manual / gated:
# - MIT/Stanford Severson 2019 LFP:
#   https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
#   Download one batch manually, preferably:
#   2017-05-12_batchdata_updated_struct_errorcorrect.mat
# - Sandia/HNEI Battery Archive exports require email access from info@batteryarchive.org.
# - NASA Open Data metadata exists at https://data.nasa.gov/dataset/li-ion-battery-aging-datasets,
#   but the CKAN package currently exposes zero downloadable resources; use manual DASHlink/NASA
#   files if NASA fallback is needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
CALCE_DIR="${RAW_DIR}/calce"
NASA_DIR="${RAW_DIR}/nasa"
SANDIA_DIR="${RAW_DIR}/sandia"
MIT_DIR="${RAW_DIR}/mit_stanford"

mkdir -p "${CALCE_DIR}" "${NASA_DIR}" "${SANDIA_DIR}" "${MIT_DIR}"

download_once() {
  local url="$1"
  local target="$2"
  if [[ -s "${target}" ]]; then
    echo "exists: ${target} ($(du -h "${target}" | awk '{print $1}'))"
    return 0
  fi
  echo "downloading: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 2 --connect-timeout 20 -o "${target}" "${url}" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${target}" "${url}" || return 1
  else
    echo "Neither curl nor wget is installed."
    return 1
  fi
}

echo "== CALCE CS2/CX2 LCO =="
CALCE_CELLS=(
  CS2_33 CS2_34 CS2_35 CS2_36 CS2_37 CS2_38
  CX2_33 CX2_34 CX2_35 CX2_36 CX2_37 CX2_38
)
for cell in "${CALCE_CELLS[@]}"; do
  url="https://web.calce.umd.edu/batteries/data/${cell}.zip"
  target="${CALCE_DIR}/${cell}.zip"
  if ! download_once "${url}" "${target}"; then
    echo "CALCE download failed: ${url}"
    echo "Manual step: open https://calce.umd.edu/battery-data and download ${cell} into ${CALCE_DIR}."
    rm -f "${target}"
  fi
done

echo "== NASA PCoE fallback =="
if compgen -G "${NASA_DIR}/B*.mat" >/dev/null; then
  echo "NASA .mat files already present in ${NASA_DIR}"
else
  echo "NASA Open Data page is public but currently lists no downloadable resources."
  echo "Manual step: place B0005.mat, B0006.mat, B0007.mat, and B0018.mat in ${NASA_DIR} if needed."
fi

echo "== MIT/Stanford Severson LFP =="
if compgen -G "${MIT_DIR}/*.mat" >/dev/null; then
  echo "MIT/Stanford .mat files already present in ${MIT_DIR}"
else
  echo "Manual step required: open https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
  echo "Download one batch, preferably 2017-05-12_batchdata_updated_struct_errorcorrect.mat."
  echo "Place it in ${MIT_DIR}/"
fi

echo "== Sandia / HNEI Battery Archive =="
echo "Email info@batteryarchive.org now."
echo "Subject: Request: SNL cycling dataset CSVs for academic project"
echo "Ask for SNL Battery Archive CSV exports: *_cycle_data.csv and *_timeseries.csv."

echo "== Raw data summary =="
for dataset_dir in "${RAW_DIR}"/*; do
  [[ -d "${dataset_dir}" ]] || continue
  count="$(find "${dataset_dir}" -type f ! -name '.gitkeep' | wc -l | tr -d ' ')"
  size="$(du -sh "${dataset_dir}" | awk '{print $1}')"
  echo "$(basename "${dataset_dir}"): ${count} files, ${size}"
done
echo "total raw disk usage: $(du -sh "${RAW_DIR}" | awk '{print $1}')"

