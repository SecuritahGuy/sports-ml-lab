# RALPH Loop 6: Five Focused Challengers

*Incumbent: v3.0.0 Frozen QB Overlay (holdout LL 0.62)*

## Challengers

| ID | Description | Hypothesis |
|----|-------------|-----------|
| prior_win_pct | +prior-season win% | Early weeks need better priors |
| weather_missing | +weather_missing+is_dome+outdoor | Missing weather flag adds signal |
| roof_enc | +roof_enc | Roof type corrects dome/retractable bias |
| games_since_change | +games_since_qb_change | QB continuity beyond binary changed flag |
| isotonic | Isotonic instead of Platt | Better calibration for all probabilities |

## Validation (Rolling-Origin)

| Model | Avg Val LL | Fold1 | Fold2 | Fold3 | Δ vs Inc |
|-------|-----------|-------|-------|-------|----------|
| incumbent | 0.6305 | 0.6360 | 0.6596 | 0.5960 | +0.0000 |
| prior_win_pct | 0.6316 | 0.6360 | 0.6627 | 0.5960 | +0.0011 |
| weather_missing | 0.6481 | 0.6846 | 0.6607 | 0.5991 | +0.0176 |
| roof_enc | 0.6315 | 0.6301 | 0.6643 | 0.6000 | +0.0010 |
| games_since_change | 0.6321 | 0.6291 | 0.6691 | 0.5981 | +0.0016 |
| isotonic | 0.6312 | 0.6417 | 0.6634 | 0.5885 | +0.0007 |

## Holdout (2025)

| Model | Hold LL | Brier | Acc | AUC | Δ vs Inc |
|-------|---------|-------|-----|-----|----------|
| incumbent | 0.62 | 0.2157 | 0.663 | 0.7098 | +0.0000 |
| prior_win_pct | 0.6183 | 0.2152 | 0.663 | 0.7112 | -0.0017 |
| weather_missing | 0.6198 | 0.2156 | 0.6522 | 0.712 | -0.0002 |
| roof_enc | 0.6202 | 0.2157 | 0.6522 | 0.7121 | +0.0002 |
| games_since_change | 0.6202 | 0.2162 | 0.6522 | 0.7081 | +0.0002 |
| isotonic | 0.6283 | 0.2191 | 0.6558 | 0.7142 | +0.0083 |

## Subgroup Impact (2025 Holdout)

| Subgroup | Model | N | Log Loss |
|----------|-------|---|----------|
| Early season (Weeks 1-4) | incumbent | 61 | 0.5844 |
| Early season (Weeks 1-4) | prior_win_pct | 61 | 0.5888 |
| Early season (Weeks 1-4) | weather_missing | 61 | 0.5818 |
| Early season (Weeks 1-4) | roof_enc | 61 | 0.5841 |
| Early season (Weeks 1-4) | games_since_change | 61 | 0.5910 |
| Early season (Weeks 1-4) | isotonic | 61 | 0.5859 |
| Weather data missing | incumbent | 92 | 0.6452 |
| Weather data missing | prior_win_pct | 92 | 0.6465 |
| Weather data missing | weather_missing | 92 | 0.6438 |
| Weather data missing | roof_enc | 92 | 0.6451 |
| Weather data missing | games_since_change | 92 | 0.6493 |
| Weather data missing | isotonic | 92 | 0.6414 |
| Retractable/open roof | incumbent | 0 | — |
| Retractable/open roof | prior_win_pct | 0 | — |
| Retractable/open roof | weather_missing | 0 | — |
| Retractable/open roof | roof_enc | 0 | — |
| Retractable/open roof | games_since_change | 0 | — |
| Retractable/open roof | isotonic | 0 | — |
| QB changed | incumbent | 55 | 0.6674 |
| QB changed | prior_win_pct | 55 | 0.6665 |
| QB changed | weather_missing | 55 | 0.6676 |
| QB changed | roof_enc | 55 | 0.6674 |
| QB changed | games_since_change | 55 | 0.6572 |
| QB changed | isotonic | 55 | 0.6943 |

## Decisions

### ❌ prior_win_pct

Val LL: 0.6316 (Δ=0.0011), Holdout: 0.6183 (Δ=-0.0017)

Rejected — val worse (+0.0011) holdout better (-0.0017).

### ❌ weather_missing

Val LL: 0.6481 (Δ=0.0176), Holdout: 0.6198 (Δ=-0.0002)

Rejected — val worse (+0.0176) holdout better (-0.0002).

### ❌ roof_enc

Val LL: 0.6315 (Δ=0.0010), Holdout: 0.6202 (Δ=0.0002)

Rejected — val worse (+0.0010) holdout better (+0.0002).

### ❌ games_since_change

Val LL: 0.6321 (Δ=0.0016), Holdout: 0.6202 (Δ=0.0002)

Rejected — val worse (+0.0016) holdout better (+0.0002).

### ❌ isotonic

Val LL: 0.6312 (Δ=0.0007), Holdout: 0.6283 (Δ=0.0083)

Rejected — val better (+0.0007) holdout worse (+0.0083).

**No challenger beats the incumbent.**

## Leakage Assessment

| Feature | Source | Leakage Risk | Live-Safe |
|---------|--------|--------------|-----------|
| prior_win_pct | Prior season results | None (available precseason) | Yes |
| weather_missing_flag | Feature table | None | Yes |
| is_dome | Stadium info | None | Yes |
| outdoor_game_flag | Stadium info | None | Yes |
| roof_enc | Stadium info | None | Yes |
| games_since_qb_change | Chronological QB tracker | None (pregame) | Yes (weekly tracker) |

## Next Steps

1. All challengers rejected — incumbent unchanged
2. Preserve challenger code for future comparison
3. Consider testing with 2026 data when available

