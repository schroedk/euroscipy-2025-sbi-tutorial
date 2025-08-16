#!/usr/bin/env python3
"""Setup test for EuroSciPy 2025 SBI Tutorial"""


def main():
    print("SBI Tutorial - Environment Check")
    print("=" * 35)

    try:
        import torch  # noqa: F401
        from sbi.utils import BoxUniform

        # Test basic functionality
        BoxUniform(low=torch.tensor([0.0]), high=torch.tensor([1.0]))

        print("✅ All packages working")
        print("🎉 Ready for tutorial!")

    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: uv pip install -e .")


if __name__ == "__main__":
    main()
