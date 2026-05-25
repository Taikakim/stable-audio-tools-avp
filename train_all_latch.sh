#!/bin/bash
# train_all_latch.sh
# Trains LatCH models for all features except tonic and spectral_flatness

features=(
    "beat_activations"
    "downbeat_activations"
    "hpcp"
    "onsets_activations"
    "rms_energy_air"
    "rms_energy_bass"
    "rms_energy_body"
    "rms_energy_mid"
    "spectral_flux"
    "spectral_kurtosis"
    "spectral_skewness"
    "tonic_strength"
)

for feature in "${features[@]}"; do
    echo "=========================================================="
    echo "Training LatCH model for feature: $feature"
    echo "=========================================================="
    python3 scripts/train_latch.py --feature "$feature" --epochs 10
    
    if [ $? -ne 0 ]; then
        echo "Error training $feature. Moving to next..."
    fi
done

echo "All training loops completed."
