# Gordon + Neural Memory Experiment

Test Docker's Gordon AI with and without Neural Memory enhancement.

## The Hypothesis

**Without Neural Memory:**
- Gordon asks clarifying questions each session
- No memory of user preferences across sessions
- Each interaction starts from scratch

**With Neural Memory:**
- Learns user's Docker preferences (base images, Python version, compose patterns)
- Fewer clarifying questions over time
- Surprise score drops as patterns become familiar

## Quick Start

```bash
# Ensure Docker Desktop is running with Gordon enabled
# Settings → Features in development → Enable Docker AI

# Run the automated test
cd experiments/gordon
python test_gordon.py
```

## Test Scenarios

The test runs these Docker-focused queries in sequence:

1. **"Create a Dockerfile for a Python web application"**
   - Baseline - Gordon will ask about Python version, base image

2. **"Add a multi-stage build to optimize the image size"**
   - With memory: Should remember Python version
   - Without: May ask again

3. **"Create a docker-compose.yml for this app with Redis"**
   - Tests if it remembers the app context

4. **"What's the best way to handle secrets in this setup?"**
   - Tests generalization to related topics

5. **"Optimize this Dockerfile for production"**
   - Should now have full context of preferences

## Expected Results

| Metric | Without Memory | With Memory |
|--------|----------------|-------------|
| Clarifying questions | 4-5 | 1-2 |
| Surprise (start) | N/A | ~0.8 |
| Surprise (end) | N/A | ~0.2 |
| Preferences remembered | 0 | All |

## Manual Testing

You can also test manually:

```bash
# Without memory context
docker ai "Create a Dockerfile for my Python app"

# With memory context (paste this)
docker ai "Context: User prefers Python 3.11, slim base images, Docker Compose v3.

User question: Create a Dockerfile for my Python app"
```

Compare the responses!

## Metrics Collected

- **Clarification rate**: Did Gordon ask for more info?
- **Surprise score**: How novel was this query to neural memory?
- **Preferences learned**: What did we learn about the user?
- **Response time**: Any latency differences?

## Files

- `test_gordon.py` - Automated comparison test
- `gordon_test_results.json` - Test results (generated)
