Log–log slope = linear fit of **sc** (or **tc**) vs **log₂ χ**. Since sc = log₂(largest intermediate size), slope **k** means peak space scales as **χ^k** (similarly for tc vs total FLOPs).

BP (§1) and TNMP rank-2 **3×1 cavity / 3×3 neighborhood** (L = 3, §3), TreeSA.

### sc slopes

| Method | Line | χ range | Slope |
|--------|------|---------|------:|
| BP | cavity | {4, 8, 16, 32} | 4.000 |
| BP | neighborhood | {4, 8, 16, 32} | 4.000 |
| TNMP | rank2 cavity (3×1) | {4, 8, 16, 32} | 4.000 |
| TNMP | rank2 neighborhood (3×3) | {4, 8, 16, 32} | 7.400 |

### tc slopes

| Method | Line | χ range | Slope |
|--------|------|---------|------:|
| BP | cavity | {4, 8, 16, 32} | 5.000 |
| BP | neighborhood | {4, 8, 16, 32} | 4.999 |
| TNMP | rank2 cavity (3×1) | {4, 8, 16, 32} | 5.000 |
| TNMP | rank2 neighborhood (3×3) | {4, 8, 16, 32} | 12.182 |
