"""Test that benefits queries don't expose employee risk data"""
import sys
for mod in list(sys.modules.keys()):
    if 'app' in mod:
        del sys.modules[mod]

from app.agents.orchestrator import detect_agent, route

# Test 1: Policy query detection
query = 'What benefits does the company offer?'
agent = detect_agent(query)
print('Test 1: Intent Detection')
print('='*80)
print(f'Query: {query}')
print(f'Detected Agent: {agent}')
print()

# Test 2: Manager routing for policy query (should NOT include risk data)
print('Test 2: Manager Query About Benefits for Employee #4')
print('='*80)
result = route(query, employee_id=4, caller_role='manager')
print(f'Agent: {result["agent"]}')
print(f'Response keys: {list(result["result"].keys())}')
print()
print('Full Response:')
import json
print(json.dumps(result['result'], indent=2, default=str)[:400])
print()

# Check if attrition_probability is in response (it shouldn't be for policy queries)
result_str = str(result['result'])
if 'attrition_probability' in result_str or '0.796' in result_str:
    print('❌ PROBLEM: attrition_probability is still in response!')
else:
    print('✅ FIXED: attrition_probability NOT in response (policy-only)')

print()
print('Test 3: Verify policy content is correct')
print('='*80)
if 'answer' in result['result'] and 'benefits' in result['result']['answer'].lower():
    print('✅ Policy answer includes benefits information')
else:
    print('❌ Policy answer missing')
