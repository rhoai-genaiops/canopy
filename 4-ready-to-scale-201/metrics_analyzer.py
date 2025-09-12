import yaml
from pathlib import Path

def analyze_benchmark_results(output_path):
    """
    Analyze GuideLLM benchmark results and provide performance summary
    """
    # Load and analyze benchmark results
    results_file = Path(output_path)
    if results_file.exists():
        with open(results_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract key metrics from the benchmark results
        benchmark = data['benchmarks'][0]
        metrics = benchmark['metrics']
        totals = benchmark['request_totals']
        
        print("🎯 CANOPY BACKEND PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Success Rate
        success_rate = (totals['successful'] / totals['total']) * 100
        print(f"📊 Success Rate: {success_rate:.1f}% ({totals['successful']}/{totals['total']} requests)")
        
        # Time to First Token (TTFT)
        ttft_median = metrics['time_to_first_token_ms']['successful']['median']
        ttft_status = "🟢 Excellent" if ttft_median < 200 else "🟡 Good" if ttft_median < 500 else "🔴 Needs Improvement"
        print(f"🚀 Time to First Token: {ttft_median:.0f}ms {ttft_status}")
        
        # Output Tokens per Second
        tokens_per_sec = metrics['output_tokens_per_second']['successful']['median']
        tps_status = "🟢 Great" if tokens_per_sec > 30 else "🟡 Acceptable" if tokens_per_sec > 20 else "🔴 Slow"
        print(f"⚡ Output Tokens/Second: {tokens_per_sec:.1f} {tps_status}")
        
        # Request Latency
        req_latency = metrics['request_latency']['successful']['mean']
        latency_status = "🟢 Fast" if req_latency < 10 else "🟡 Acceptable" if req_latency < 20 else "🔴 Slow"
        print(f"🎯 Request Latency: {req_latency:.1f}s {latency_status}")
        
        # Requests per Second
        rps = metrics['requests_per_second']['successful']['mean']
        rps_status = "🟢 High" if rps > 1 else "🟡 Moderate" if rps > 0.1 else "🔴 Low"
        print(f"📈 Requests/Second: {rps:.3f} {rps_status}")
        
        # Inter-Token Latency (consistency)
        inter_token = metrics['inter_token_latency_ms']['successful']['mean']
        consistency_status = "🟢 Smooth" if inter_token < 30 else "🟡 Acceptable" if inter_token < 50 else "🔴 Choppy"
        print(f"⚙️  Inter-Token Latency: {inter_token:.1f}ms {consistency_status}")
        
        # Output Quality
        avg_output_tokens = metrics['output_token_count']['successful']['mean']
        output_status = "🟢 Rich" if avg_output_tokens > 300 else "🟡 Moderate" if avg_output_tokens > 100 else "🔴 Short"
        print(f"📝 Average Response Length: {avg_output_tokens:.0f} tokens {output_status}")
        
        print("\n" + "=" * 50)
        print("🏆 OVERALL ASSESSMENT")
        print("=" * 50)
        
        # Overall score calculation
        scores = []
        scores.append(100 if success_rate > 95 else 80 if success_rate > 80 else 60)
        scores.append(100 if ttft_median < 200 else 80 if ttft_median < 500 else 60)
        scores.append(100 if tokens_per_sec > 30 else 80 if tokens_per_sec > 20 else 60)
        scores.append(100 if req_latency < 10 else 80 if req_latency < 20 else 60)
        scores.append(100 if inter_token < 30 else 80 if inter_token < 50 else 60)
        
        overall_score = sum(scores) / len(scores)
        
        if overall_score >= 90:
            verdict = "🏆 EXCELLENT - Production Ready!"
        elif overall_score >= 75:
            verdict = "🟢 GOOD - Ready with minor optimizations"
        elif overall_score >= 60:
            verdict = "🟡 ACCEPTABLE - Needs optimization"
        else:
            verdict = "🔴 POOR - Requires significant improvement"
        
        print(f"Overall Performance Score: {overall_score:.0f}/100")
        print(f"Verdict: {verdict}")
        
        print("\n💡 Key Takeaways:")
        if ttft_median < 200:
            print("   ✅ Excellent responsiveness - users get immediate feedback")
        if tokens_per_sec > 25:
            print("   ✅ Good generation speed - comfortable reading pace")
        if success_rate >= 80:
            print("   ✅ Reliable performance - suitable for production")
        if inter_token < 35:
            print("   ✅ Smooth streaming - consistent user experience")
        
        if success_rate < 95:
            print("   ⚠️  Consider timeout tuning to improve success rate")
        if rps < 0.1:
            print("   ⚠️  Test concurrent requests to measure true throughput capacity")
            
    else:
        print(f"❌ Results file not found: {results_file}")
        print("   Make sure the benchmark completed successfully.")