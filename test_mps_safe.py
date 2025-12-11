#!/usr/bin/env python3
"""
Safe MPS availability check with proper error handling
Matches the pattern used in src/finetuner.py
"""
import json
import sys
import platform
from datetime import datetime

LOG_PATH = "/Users/gaurav/codebase/simplesql/.cursor/debug.log"

def log_debug(session_id, run_id, hypothesis_id, location, message, data=None):
    """Write debug log entry"""
    # #region agent log
    try:
        with open(LOG_PATH, 'a') as f:
            log_entry = {
                "id": f"log_{int(datetime.now().timestamp() * 1000)}",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "sessionId": session_id,
                "runId": run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data or {}
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}", file=sys.stderr)
    # #endregion

def check_mps_availability():
    """
    Safely check MPS availability using the same pattern as finetuner.py
    Returns: bool or None if check fails
    """
    session_id = "debug-session"
    run_id = "run1"
    
    try:
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "Starting MPS check")
        # #endregion
        
        import torch
        
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "Torch imported", {
            "torch_version": torch.__version__
        })
        # #endregion
        
        # Use the same safe pattern as finetuner.py
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "Before hasattr check")
        # #endregion
        
        if hasattr(torch.backends, 'mps'):
            # #region agent log
            log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "MPS backend exists, calling is_available()")
            # #endregion
            
            result = torch.backends.mps.is_available()
            
            # #region agent log
            log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "is_available() completed", {
                "result": result
            })
            # #endregion
            
            return result
        else:
            # #region agent log
            log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "MPS backend not found", {
                "torch_version": torch.__version__
            })
            # #endregion
            
            return False
            
    except ImportError as e:
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "ImportError", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        # #endregion
        
        print(f"ERROR: Failed to import torch: {e}")
        print("Install PyTorch with: pip install torch torchvision torchaudio")
        return None
        
    except AttributeError as e:
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "AttributeError", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        # #endregion
        
        print(f"ERROR: Attribute error - {e}")
        print("This PyTorch version may not support MPS backend")
        return None
        
    except Exception as e:
        # #region agent log
        log_debug(session_id, run_id, "SAFE", "test_mps_safe.py:check_mps_availability", "Unexpected error", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        # #endregion
        
        print(f"ERROR: Unexpected error: {e}")
        return None

if __name__ == "__main__":
    result = check_mps_availability()
    if result is not None:
        print(f"MPS Available: {result}")
    else:
        print("MPS check failed - see errors above")
        sys.exit(1)
