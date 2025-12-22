# Wandb Configuration Guide for LOMO

## Problem
LOMO code would prompt for interactive Wandb login/selection when running, requiring user to manually choose from:
1. Create a new account
2. Use existing account  
3. Don't visualize my results

## Solution
Added Wandb configuration options to YAML files to avoid interactive prompts.

## Configuration Options

### In YAML Config File

```yaml
# Wandb configuration
wandb_mode: 'online'          # Options: 'online', 'offline', 'disabled'
wandb_project: 'your-project' # Wandb project name
wandb_entity: 'your-entity'   # Your wandb username/team name
wandb_run_name: null          # Optional: custom run name (auto-generated if null)
```

## Usage Examples

### 1. Completely Disable Wandb (No Tracking)
```yaml
wandb_mode: 'disabled'
# wandb_project and wandb_entity not needed when disabled
```

### 2. Offline Mode (Track Locally, Sync Later)
```yaml
wandb_mode: 'offline'
wandb_project: 'lomo-training'
wandb_entity: 'your-username'
```

### 3. Online Mode (Default - Track in Real-time)
```yaml
wandb_mode: 'online'
wandb_project: 'lomo-training'
wandb_entity: 'your-username'
```

## Quick Start

1. **For No Wandb Tracking:**
   ```bash
   python src/train_lomo.py config/args_lomo_no_wandb.yaml
   ```

2. **For Wandb Tracking:**
   - First, login to wandb: `wandb login`
   - Edit `wandb_entity` in config file to your username
   - Run: `python src/train_lomo.py config/args_lomo.yaml`

## Environment Variables Set Automatically

The code automatically sets these environment variables to prevent prompts:
- `WANDB_MODE`: Controls wandb mode
- `WANDB_SILENT`: Prevents login prompts

## Files Modified

1. `src/arguments.py` - Added wandb configuration fields
2. `src/train_lomo.py` - Updated wandb initialization logic
3. `src/train_lomo_continued_pretraining.py` - Updated wandb initialization logic
4. `config/args_lomo.yaml` - Added wandb configuration
5. `config/args_continued_pretraining.yaml` - Added wandb configuration
6. `config/args_lomo_no_wandb.yaml` - New config with wandb disabled

## Benefits

- ✅ No more interactive prompts
- ✅ Configurable via YAML files
- ✅ Can completely disable wandb if not needed
- ✅ Can run in offline mode for limited internet environments
- ✅ Backward compatible with existing configs