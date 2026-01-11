# Logging and Results Tracking - Summary

This document provides a summary of the new logging and results tracking functionality added to the explainy project.

## Overview

Two new features have been implemented:
1. **Results tracking** - Stores processing results for each topic in JSON format (rewrite mode)
2. **Logging** - Records detailed processing information in a log file (append mode)

## Files Created

### 1. `outputs/results.json`
**Purpose**: Track the current status of all processed topics

**Format**: JSON file with topic codes as keys

**Example**:
```json
{
  "prf": {
    "model": "gemini-3-preview",
    "status": "success",
    "last_updated": "2026-01-11T15:13:52.192667"
  },
  "dpr": {
    "model": "gemini-3-preview",
    "status": "failed",
    "last_updated": "2026-01-11T15:13:52.193282"
  }
}
```

**Fields**:
- `model`: LLM model used for processing
- `status`: Processing status (`success` or `failed`)
- `last_updated`: ISO 8601 timestamp of last update

**Mode**: File is rewritten on each run to reflect current state

### 2. `outputs/processing.log`
**Purpose**: Maintain a complete history of all topic processing attempts

**Format**: Tab-separated values (TSV)

**Example**:
```
2026-01-11T15:13:52.192680	Pseudo-Relevance Feedback	gemini-3-preview	1234	success
2026-01-11T15:13:52.193290	Deep Passage Retrieval	gemini-3-preview	2345	success
2026-01-11T15:13:52.193377	ColBERT	gemini-3-preview	0	failed
```

**Fields** (in order):
1. Timestamp (ISO 8601)
2. Topic name
3. Model used
4. Token count
5. Status (`success` or `failed`)

**Mode**: New entries are appended, preserving full history

## Implementation Details

### Functions Added to `main.py`

1. **`save_results(filepath, results_data)`**
   - Saves results dictionary to JSON file
   - Overwrites existing file (rewrite mode)
   - Returns: `bool` (success/failure)

2. **`load_results(filepath)`**
   - Loads results from JSON file
   - Returns empty dict if file doesn't exist
   - Returns: `dict` with results

3. **`update_result(results_data, topic_code, model, status)`**
   - Updates a single topic's result in memory
   - Validates status parameter
   - Updates timestamp automatically

4. **`log_processing(log_filepath, topic, model, tokens, status)`**
   - Appends processing information to log file
   - Validates status parameter
   - Returns: `bool` (success/failure)

### Modified Functions

**`generate_explanation()`**
- **Before**: Returned only the explanation text
- **After**: Returns tuple of `(explanation, token_count)`
- This allows tracking token usage for each topic

### Integration in Main Loop

```python
# Load existing results at startup
results_data = load_results(results_filepath)

# Process each topic
for topic_data in topics:
    explanation, tokens = generate_explanation(...)
    
    if explanation:
        save_complete_notebook(...)
        # Update results and log success
        update_result(results_data, code, PRIMARY_MODEL, 'success')
        log_processing(log_filepath, detailed_query, PRIMARY_MODEL, tokens, 'success')
    else:
        # Log failure
        update_result(results_data, code, PRIMARY_MODEL, 'failed')
        log_processing(log_filepath, detailed_query, PRIMARY_MODEL, 0, 'failed')

# Save updated results at the end
save_results(results_filepath, results_data)
```

## Testing

Comprehensive tests have been added in `tests/test_logging_and_results.py`:

1. **test_save_and_load_results**: Tests saving and loading JSON results
2. **test_update_result**: Tests updating results dictionary
3. **test_log_processing**: Tests appending to log file
4. **test_load_nonexistent_file**: Tests handling of missing files
5. **test_status_validation**: Tests validation of status values

Demo script: `tests/demo_logging_and_results.py` shows realistic usage

## Benefits

1. **Track Progress**: See which topics succeeded/failed at a glance
2. **Monitor Token Usage**: Track API costs via token counts in logs
3. **Debug Issues**: Full history in log file helps identify problems
4. **Resume Processing**: Results file shows what needs to be retried
5. **Audit Trail**: Complete history of all processing attempts

## Usage Example

After running the main application:

```bash
# View current status of all topics
cat outputs/results.json | jq

# View processing history
cat outputs/processing.log

# Find failed topics
cat outputs/processing.log | grep failed

# Calculate total tokens used
cat outputs/processing.log | awk '{sum += $4} END {print sum}'
```

## Type Safety

All new functions use proper type hints:
- `Union[str, Path]` for file paths (accepts both types)
- Status validation ensures only 'success' or 'failed' are logged
- Token counts must be integers

## Error Handling

All file operations include:
- Try-catch blocks with informative error messages
- Graceful degradation (returns False/empty dict on error)
- Status validation prevents invalid data
