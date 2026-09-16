"""
Comprehensive test suite validating all 8 model/attack combinations + regression features.
"""

import json
import urllib.request

BASE_URL = "http://127.0.0.1:5000"


def make_post(endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))


def run_tests():
    tests = [
        ("TEST 1: TF-IDF + no attack", "/predict", {
            "headline": "Scientists discover a new species in the Amazon",
            "model": "tfidf"
        }),
        ("TEST 2: DistilBERT + no attack", "/predict", {
            "headline": "You won't believe what scientists discovered!",
            "model": "distilbert"
        }),
        ("TEST 3: TF-IDF + character substitution", "/adversarial", {
            "headline": "You won't believe what happened next!",
            "attack_type": "substitution",
            "model": "tfidf"
        }),
        ("TEST 4: DistilBERT + character substitution", "/adversarial", {
            "headline": "You won't believe what happened next!",
            "attack_type": "substitution",
            "model": "distilbert"
        }),
        ("TEST 5: TF-IDF + character insertion", "/adversarial", {
            "headline": "Government announces new education policy",
            "attack_type": "insertion",
            "model": "tfidf"
        }),
        ("TEST 6: DistilBERT + character insertion", "/adversarial", {
            "headline": "Government announces new education policy",
            "attack_type": "insertion",
            "model": "distilbert"
        }),
        ("TEST 7: TF-IDF + character deletion", "/adversarial", {
            "headline": "Breaking news: major changes announced today",
            "attack_type": "deletion",
            "model": "tfidf"
        }),
        ("TEST 8: DistilBERT + character deletion", "/adversarial", {
            "headline": "Breaking news: major changes announced today",
            "attack_type": "deletion",
            "model": "distilbert"
        }),
        ("TEST 9A: Compare Models Mode (Predict)", "/predict", {
            "headline": "You won't believe this shocking secret!",
            "model": "compare"
        }),
        ("TEST 9B: Compare Models Mode (Adversarial)", "/adversarial", {
            "headline": "You won't believe this shocking secret!",
            "attack_type": "substitution",
            "model": "compare"
        }),
        ("TEST 9C: Existing LIME Explainability", "/explain", {
            "headline": "You won't believe what happened next!"
        })
    ]

    results = []
    print("=" * 60)
    print("RUNNING VALIDATION SUITE")
    print("=" * 60)
    for name, endpoint, payload in tests:
        try:
            res = make_post(endpoint, payload)
            print(f"\n[PASS] {name}")
            print(json.dumps(res, indent=2))
            results.append((name, True, res))
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = all(r[1] for r in results)
    for name, ok, _ in results:
        status = "PASSED" if ok else "FAILED"
        print(f"{name}: {status}")
    print(f"\nOverall: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED'}")


if __name__ == "__main__":
    run_tests()
