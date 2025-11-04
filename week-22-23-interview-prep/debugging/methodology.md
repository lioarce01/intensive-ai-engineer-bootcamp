# Debugging Production AI Systems - Methodology

## 🎯 Overview

Debugging production AI systems requires a **systematic approach** that differs from traditional software debugging. You must consider:
- Model behavior (non-deterministic)
- Data quality issues
- Infrastructure problems
- Cost and latency constraints

## 🔍 The 6-Step Debugging Framework

```
1. OBSERVE   → What are the symptoms?
2. HYPOTHESIZE → What could cause this?
3. TEST      → How can we verify?
4. ANALYZE   → What does the data show?
5. FIX       → Implement solution
6. PREVENT   → How to avoid in future?
```

## 📋 Step 1: OBSERVE - Gather Information

### Initial Questions

When a production issue is reported, ask:

**Impact Assessment**:
- ✅ How many users are affected? (1 user vs 1000 users)
- ✅ What's the severity? (P0: service down, P1: degraded, P2: minor)
- ✅ When did it start? (timestamp matters for correlation)
- ✅ Is it reproducible? (always, sometimes, random?)

**Symptoms**:
- ✅ What's the observed behavior vs expected behavior?
- ✅ Are there error messages? (collect full stack traces)
- ✅ What's the user flow when it happens?

**Environment**:
- ✅ Production, staging, or development?
- ✅ Which region/datacenter?
- ✅ Recent deployments or changes?

### Data Collection Checklist

```bash
# 1. Application logs
kubectl logs <pod-name> --since=1h --tail=1000

# 2. Error rates
curl -X POST "https://api.datadog.com/api/v1/query" \
  -d "query=sum:error_rate{env:prod}"

# 3. Latency metrics
curl -X POST "https://api.datadog.com/api/v1/query" \
  -d "query=avg:latency_p95{service:api}"

# 4. Infrastructure metrics
kubectl top pods
kubectl top nodes

# 5. Recent deployments
kubectl rollout history deployment/api-service

# 6. Recent scaling events
kubectl get hpa api-service -o yaml
```

### Observation Template

```markdown
## Issue Report

**Incident ID**: INC-2024-001
**Reporter**: user@example.com
**Severity**: P1 (service degraded)
**Start Time**: 2024-01-15 14:30 UTC

### Symptoms
- API latency increased from 200ms to 3s (P95)
- Error rate: 0% → 5%
- Affected users: ~1000 (10% of active users)

### Timeline
- 14:30 - Latency spike begins
- 14:35 - Error rate increases
- 14:40 - On-call engineer paged

### Environment
- Production (us-east-1)
- Recent deployment: Yes (14:15 - new model version)
- Infrastructure changes: No

### Initial Observations
- Model service CPU at 95%
- Memory usage normal
- Database queries normal
- Load balancer showing 503 errors
```

## 💡 Step 2: HYPOTHESIZE - Generate Theories

### Common Failure Modes in AI Systems

| Category | Possible Causes | Quick Test |
|----------|----------------|------------|
| **Latency** | Model inference slow, DB query slow, network issue | Profile critical path |
| **Quality** | Model drift, prompt issue, data problem | Check recent predictions |
| **Availability** | Service down, rate limit hit, OOM | Check health endpoints |
| **Cost** | Token usage spike, model change, traffic increase | Check billing metrics |

### Hypothesis Framework

For each hypothesis, document:

1. **Theory**: What could be wrong?
2. **Evidence**: What supports this theory?
3. **Test**: How can we verify?
4. **Priority**: High/Medium/Low (based on likelihood × impact)

**Example**:

```markdown
### Hypothesis 1: New model is slower
- **Theory**: Recent deployment of larger model causing latency
- **Evidence**:
  - Deployment at 14:15, issue at 14:30 (15 min gap)
  - Model changed from 7B to 13B parameters
  - CPU usage at 95%
- **Test**: Check model inference time in logs
- **Priority**: HIGH (timing matches, clear evidence)

### Hypothesis 2: Database connection pool exhausted
- **Theory**: Too many concurrent requests, pool exhausted
- **Evidence**:
  - Some 503 errors (connection timeout)
  - But DB metrics show normal
- **Test**: Check connection pool stats
- **Priority**: MEDIUM (partial evidence)

### Hypothesis 3: DDoS attack
- **Theory**: Unusual traffic spike
- **Evidence**:
  - Traffic looks normal in metrics
  - Error rate only 5% (would be higher in DDoS)
- **Test**: Check request patterns
- **Priority**: LOW (evidence doesn't support)
```

## 🧪 Step 3: TEST - Verify Hypotheses

### Testing Strategies

#### 1. Log Analysis

```python
import re
from collections import Counter
from datetime import datetime

def analyze_logs(log_file: str):
    """
    Parse logs to find patterns
    """
    latencies = []
    errors = Counter()
    timestamps = []

    with open(log_file) as f:
        for line in f:
            # Extract latency
            if "latency_ms" in line:
                match = re.search(r"latency_ms=(\d+)", line)
                if match:
                    latencies.append(int(match.group(1)))

            # Extract errors
            if "ERROR" in line:
                match = re.search(r"ERROR: (.+)", line)
                if match:
                    errors[match.group(1)] += 1

            # Extract timestamp
            match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                timestamps.append(match.group(1))

    print(f"Median latency: {np.median(latencies)}ms")
    print(f"P95 latency: {np.percentile(latencies, 95)}ms")
    print(f"P99 latency: {np.percentile(latencies, 99)}ms")
    print(f"\nTop errors:")
    for error, count in errors.most_common(5):
        print(f"  {error}: {count}")
```

#### 2. Distributed Tracing

```python
from opentelemetry import trace
import time

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("api_request")
def handle_request(request):
    span = trace.get_current_span()
    span.set_attribute("request_id", request.id)

    start = time.time()

    # Each step instrumented
    with tracer.start_as_current_span("validate_input"):
        validate(request)

    with tracer.start_as_current_span("model_inference"):
        result = model.predict(request.input)
        span.set_attribute("model_latency_ms", (time.time() - start) * 1000)

    with tracer.start_as_current_span("postprocess"):
        response = postprocess(result)

    return response
```

Then view traces in Jaeger/DataDog to identify bottleneck.

#### 3. A/B Testing (Gradual Rollout)

```python
class ModelRouter:
    """
    Route traffic between old and new model for testing
    """
    def __init__(self):
        self.old_model = load_model("v1.0")
        self.new_model = load_model("v1.1")
        self.rollout_percentage = 10  # Start with 10%

    def predict(self, input_data, user_id):
        # Hash-based routing (consistent per user)
        if hash(user_id) % 100 < self.rollout_percentage:
            return self.new_model.predict(input_data)
        else:
            return self.old_model.predict(input_data)
```

**Metrics to compare**:
- Latency (P50, P95, P99)
- Error rate
- Quality (user feedback, eval metrics)
- Cost (tokens used, compute time)

#### 4. Synthetic Testing

```python
import asyncio
import time

async def load_test(endpoint: str, num_requests: int):
    """
    Send synthetic requests to test behavior
    """
    results = []

    async def single_request():
        start = time.time()
        try:
            response = await client.post(endpoint, json={"query": "test"})
            latency = (time.time() - start) * 1000
            return {"status": "success", "latency_ms": latency}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # Fire requests concurrently
    results = await asyncio.gather(*[
        single_request() for _ in range(num_requests)
    ])

    # Analyze results
    latencies = [r["latency_ms"] for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"Success rate: {len(latencies) / len(results) * 100:.1f}%")
    print(f"Median latency: {np.median(latencies):.0f}ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.0f}ms")
    print(f"Errors: {len(errors)}")
```

## 📊 Step 4: ANALYZE - Interpret Data

### Analysis Techniques

#### 1. Time Series Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_time_series(metrics_df: pd.DataFrame):
    """
    Plot metrics over time to identify patterns
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Latency over time
    axes[0].plot(metrics_df['timestamp'], metrics_df['latency_p95'])
    axes[0].axvline(x=deployment_time, color='r', linestyle='--', label='Deployment')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].legend()

    # Error rate over time
    axes[1].plot(metrics_df['timestamp'], metrics_df['error_rate'])
    axes[1].axvline(x=deployment_time, color='r', linestyle='--')
    axes[1].set_ylabel('Error Rate (%)')

    # Throughput over time
    axes[2].plot(metrics_df['timestamp'], metrics_df['requests_per_second'])
    axes[2].axvline(x=deployment_time, color='r', linestyle='--')
    axes[2].set_ylabel('RPS')

    plt.tight_layout()
    plt.savefig('incident_analysis.png')
```

#### 2. Correlation Analysis

```python
def find_correlations(metrics_df: pd.DataFrame):
    """
    Find which metrics correlate with the issue
    """
    correlation_matrix = metrics_df.corr()

    # What correlates with latency?
    latency_corr = correlation_matrix['latency_p95'].sort_values(ascending=False)

    print("Metrics correlated with latency:")
    for metric, corr in latency_corr.items():
        if metric != 'latency_p95' and abs(corr) > 0.7:
            print(f"  {metric}: {corr:.2f}")
```

**Example output**:
```
Metrics correlated with latency:
  model_inference_time: 0.95  ← Strong correlation!
  cpu_usage: 0.82
  memory_usage: 0.34
  db_query_time: 0.12  ← Weak correlation
```

#### 3. Distribution Analysis

```python
def compare_distributions(before_df, after_df, metric='latency'):
    """
    Compare metric distributions before and after incident
    """
    import scipy.stats as stats

    before_values = before_df[metric]
    after_values = after_df[metric]

    # Statistical test
    t_stat, p_value = stats.ttest_ind(before_values, after_values)

    print(f"Before: mean={before_values.mean():.1f}, std={before_values.std():.1f}")
    print(f"After:  mean={after_values.mean():.1f}, std={after_values.std():.1f}")
    print(f"Difference is statistically significant: {p_value < 0.05}")

    # Visualize
    plt.hist(before_values, bins=50, alpha=0.5, label='Before')
    plt.hist(after_values, bins=50, alpha=0.5, label='After')
    plt.legend()
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.savefig(f'{metric}_distribution.png')
```

## 🔧 Step 5: FIX - Implement Solution

### Fix Strategies by Category

#### Latency Issues

**Quick wins**:
1. Increase instance size (vertical scaling)
2. Add caching layer
3. Reduce batch size (if batching)

**Long-term**:
1. Model optimization (quantization, distillation)
2. Infrastructure upgrades
3. Code optimization (profiling)

**Example rollback script**:
```bash
#!/bin/bash
# Rollback to previous deployment

# 1. Identify previous version
PREV_REVISION=$(kubectl rollout history deployment/api-service | tail -n 2 | head -n 1 | awk '{print $1}')

# 2. Rollback
kubectl rollout undo deployment/api-service --to-revision=$PREV_REVISION

# 3. Wait for rollout
kubectl rollout status deployment/api-service

# 4. Verify
curl https://api.example.com/health
```

#### Quality Issues

**Quick wins**:
1. Revert to previous model
2. Adjust temperature (lower = more deterministic)
3. Update system prompt

**Long-term**:
1. Retrain/fine-tune model
2. Improve evaluation pipeline
3. Add quality gates

#### Cost Issues

**Quick wins**:
1. Add aggressive caching
2. Use smaller model for simple queries
3. Set max_tokens limit

**Long-term**:
1. Optimize prompts (fewer tokens)
2. Batch processing where possible
3. Model distillation

## 🛡️ Step 6: PREVENT - Future-Proofing

### Prevention Checklist

#### 1. Monitoring & Alerts

```yaml
# alerts.yaml
alerts:
  - name: HighLatency
    query: "avg:api.latency.p95{env:prod} > 500"
    message: "API latency P95 > 500ms"
    severity: warning
    notify: [email, slack, pagerduty]

  - name: CriticalLatency
    query: "avg:api.latency.p95{env:prod} > 2000"
    message: "API latency P95 > 2s - CRITICAL"
    severity: critical
    notify: [pagerduty]

  - name: HighErrorRate
    query: "sum:api.errors{env:prod}.as_rate() > 0.01"
    message: "Error rate > 1%"
    severity: critical

  - name: ModelQualityDrop
    query: "avg:model.quality_score{env:prod} < 0.8"
    message: "Model quality score dropped below 0.8"
    severity: warning
```

#### 2. Deployment Best Practices

```python
class SafeDeployment:
    """
    Gradual rollout with automatic rollback
    """
    def __init__(self):
        self.stages = [
            {"percentage": 10, "duration_min": 30},
            {"percentage": 25, "duration_min": 60},
            {"percentage": 50, "duration_min": 120},
            {"percentage": 100, "duration_min": None}
        ]
        self.rollback_threshold = {
            "error_rate": 0.01,  # 1%
            "latency_p95": 2000,  # 2s
            "quality_score": 0.8
        }

    async def deploy(self, new_version: str):
        """
        Deploy with automatic rollback on failure
        """
        for stage in self.stages:
            print(f"Rolling out to {stage['percentage']}%...")

            # Update routing
            self.update_routing(new_version, stage['percentage'])

            # Monitor
            await asyncio.sleep(stage['duration_min'] * 60)

            # Check metrics
            metrics = await self.get_metrics(new_version)

            if self.should_rollback(metrics):
                print("Metrics exceeded threshold - rolling back!")
                self.rollback()
                return False

        print("Deployment successful!")
        return True

    def should_rollback(self, metrics: Dict) -> bool:
        for metric, threshold in self.rollback_threshold.items():
            if metrics[metric] > threshold:
                return True
        return False
```

#### 3. Testing Before Production

```python
# Pre-deployment checklist
class PreDeploymentTests:
    """
    Run before every production deployment
    """
    async def run_all_tests(self):
        results = {}

        # 1. Unit tests
        results['unit_tests'] = await self.run_unit_tests()

        # 2. Integration tests
        results['integration_tests'] = await self.run_integration_tests()

        # 3. Load tests
        results['load_tests'] = await self.run_load_tests()

        # 4. Quality tests (model evaluation)
        results['quality_tests'] = await self.run_quality_tests()

        # 5. Security tests
        results['security_tests'] = await self.run_security_tests()

        # All must pass
        return all(results.values())

    async def run_quality_tests(self):
        """
        Test model on eval dataset
        """
        eval_dataset = load_eval_dataset()

        predictions = [
            await model.predict(example['input'])
            for example in eval_dataset
        ]

        # Compute metrics
        accuracy = compute_accuracy(predictions, eval_dataset)
        latency = compute_avg_latency(predictions)

        return (
            accuracy > 0.85 and  # Quality threshold
            latency < 1000       # Latency threshold (1s)
        )
```

#### 4. Runbooks

Create runbooks for common issues:

```markdown
# Runbook: High Latency

## Symptoms
- API P95 latency > 2s
- User complaints about slow responses

## Diagnosis Steps
1. Check DataDog dashboard: https://app.datadoghq.com/dashboard/abc123
2. Look for recent deployments: `kubectl rollout history deployment/api-service`
3. Check model inference time: Query logs for "model_inference_time"
4. Check database query time: Query logs for "db_query_time"

## Resolution Steps

### If model inference is slow:
```bash
# Option 1: Rollback to previous model
kubectl rollout undo deployment/api-service

# Option 2: Scale up GPU instances
kubectl scale deployment api-service --replicas=10
```

### If database is slow:
```bash
# Check slow queries
kubectl exec -it postgres-0 -- psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Option: Add read replicas
kubectl scale statefulset postgres-read-replica --replicas=3
```

## Escalation
If issue persists > 30 min, escalate to: @ml-platform-team
```

## 📝 Debugging Worksheet Template

```markdown
## Debugging Session: [Issue Title]

### 1. OBSERVE
**Symptoms**:
-
-

**Impact**:
- Users affected:
- Severity:
- Start time:

**Data collected**:
- [ ] Logs
- [ ] Metrics
- [ ] Traces
- [ ] User reports

### 2. HYPOTHESIZE
**Hypothesis 1**: [Theory]
- Evidence:
- Test:
- Priority:

**Hypothesis 2**: [Theory]
- Evidence:
- Test:
- Priority:

### 3. TEST
**Tests run**:
- [x] Test 1: [Description] → [Result]
- [ ] Test 2: [Description] → [Pending]

### 4. ANALYZE
**Findings**:
-
-

**Root cause**:
-

### 5. FIX
**Solution implemented**:
-

**Validation**:
- [ ] Metrics returned to normal
- [ ] User confirmed fix
- [ ] No new errors

### 6. PREVENT
**Prevention measures**:
- [ ] Add monitoring
- [ ] Update runbook
- [ ] Improve testing
- [ ] Document learnings
```

---

**Next**: [Latency Issues](./latency-issues.md)
**Back**: [RAG at Scale](../rag-at-scale/architecture.md)
