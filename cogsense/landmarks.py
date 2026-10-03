"""MediaPipe 468/478 face-mesh landmark indices used by the AU core."""

from __future__ import annotations

# Subject's right side appears on the image left.
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_OUTER, L_EYE_INNER = 263, 362
R_EYE_UPPER = (160, 159, 158)
R_EYE_LOWER = (144, 145, 153)
L_EYE_UPPER = (387, 386, 385)
L_EYE_LOWER = (373, 374, 380)
# Eye-aspect-ratio point sets (p1..p6).
R_EYE_EAR = (33, 160, 158, 133, 153, 144)
L_EYE_EAR = (362, 385, 387, 263, 373, 380)

R_BROW_INNER = (107, 55)
L_BROW_INNER = (336, 285)
R_BROW_MID = (105, 66, 65)
L_BROW_MID = (334, 296, 295)
R_BROW_OUTER = (70, 46)
L_BROW_OUTER = (300, 276)

NOSE_BRIDGE = 168
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
FACE_R, FACE_L = 234, 454

MOUTH_R, MOUTH_L = 61, 291
LIP_UPPER_OUTER, LIP_LOWER_OUTER = 0, 17
LIP_UPPER_INNER, LIP_LOWER_INNER = 13, 14

NUM_LANDMARKS = 478
MIN_LANDMARKS = 468

# Lower-face points that move with speech/chewing; used by the rejection mask.
LOWER_FACE = (MOUTH_R, MOUTH_L, LIP_UPPER_OUTER, LIP_LOWER_OUTER, LIP_UPPER_INNER, LIP_LOWER_INNER, CHIN)
