# ── BDCA Configuration ────────────────────────────────────────────────────────

# Environment
MAX_STEPS                     = 250
MAX_HELD                      = 2
CWAH_ROOMS                    = ["kitchen", "livingroom", "bedroom", "bathroom"]

# LLM
LLM_MODEL                     = "gpt-4o-mini"
LLM_MAX_TOKENS                = 200
LLM_TEMPERATURE               = 0.0

# Belief graph
BELIEF_DECAY_BASE             = 0.995
BELIEF_DECAY_TEAMMATE         = 0.97
BELIEF_MIN_CONFIDENCE         = 0.03
BELIEF_OBS_CONFIDENCE         = 0.95
BELIEF_MSG_CONFIDENCE         = 0.80
BELIEF_SPREAD_FACTOR          = 0.3

# Planner weights
WEIGHT_GOAL_PROGRESS          = 1.0
WEIGHT_INFO_GAIN              = 0.8
WEIGHT_TRAVEL_COST            = 0.04
WEIGHT_REDUNDANCY             = 0.85

# Action type constants
ACT_EXPLORE                   = "explore"
ACT_CHECK                     = "check"
ACT_GRAB                      = "grab"
ACT_PUT                       = "put"
ACT_MSG                       = "message"

# Communication
COMM_STEP_COST                = 0.1
VOI_THRESHOLD                 = 0.15
COMM_COOLDOWN                 = 5
VOI_COLLISION_WEIGHT          = 0.7
VOI_ENTROPY_WEIGHT            = 0.3
MAX_MSG_LENGTH                = 500

# Partner model
PARTNER_CACHE_STEPS           = 8
COLLISION_RISK_THRESHOLD      = 0.6
PARTNER_LLM_UNCERTAINTY       = 0.5

# Planner
TOP_K_CANDIDATES              = 3

# Source labels
SRC_OBS                       = "observation"
SRC_MSG                       = "message"
SRC_INFERRED                  = "inferred"

# ── Ablation flags ────────────────────────────────────────────────────────────
USE_BELIEF_GRAPH              = True
USE_PARTNER_MODEL             = True
USE_VOI_COMM                  = True
USE_BELIEF_DECAY              = True
