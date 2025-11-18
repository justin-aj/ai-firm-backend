"""
Test Qwen3-VL client with vLLM
Verifies multimodal (image/video) inference is working correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.qwen3_vl_client import Qwen3VLClient


def test_initialization():
    """Test model initialization"""
    print("\n" + "=" * 80)
    print("1. Testing Qwen3-VL Initialization")
    print("=" * 80)
    
    try:
        print("\nLoading Qwen3-VL-8B with vLLM...")
        print("(This may take 1-2 minutes on first run)")
        
        qwen = Qwen3VLClient(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            max_model_len=32768,
            gpu_memory_utilization=0.95
        )
        
        print("✅ Qwen3-VL model loaded successfully")
        return qwen
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def test_image_analysis(qwen: Qwen3VLClient):
    """Test image understanding"""
    print("\n" + "=" * 80)
    print("2. Testing Image Analysis (OCR)")
    print("=" * 80)
    
    image_url = "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
    
    print(f"\nImage URL: {image_url}")
    print("Task: Extract text from receipt")
    print("\nGenerating response...")
    
    try:
        response = qwen.extract_text_from_image(image_url)
        
        print(f"\n✅ Extracted Text:")
        print(f"   {response[:200]}...")  # First 200 chars
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_image_question_answering(qwen: Qwen3VLClient):
    """Test image question answering"""
    print("\n" + "=" * 80)
    print("3. Testing Image Question Answering")
    print("=" * 80)
    
    image_url = "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
    question = "What is the total amount on this receipt?"
    
    print(f"\nImage: Receipt")
    print(f"Question: {question}")
    print("\nGenerating answer...")
    
    try:
        answer = qwen.analyze_image(
            image_url=image_url,
            question=question,
            temperature=0.0,
            max_tokens=256
        )
        
        print(f"\n✅ Answer:")
        print(f"   {answer}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_video_analysis(qwen: Qwen3VLClient):
    """Test video understanding"""
    print("\n" + "=" * 80)
    print("4. Testing Video Analysis")
    print("=" * 80)
    
    video_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"
    question = "What is happening in this video? Describe the key events."
    
    print(f"\nVideo URL: {video_url}")
    print(f"Question: {question}")
    print("\nGenerating answer...")
    
    try:
        answer = qwen.analyze_video(
            video_url=video_url,
            question=question,
            temperature=0.3,
            max_tokens=512
        )
        
        print(f"\n✅ Answer:")
        print(f"   {answer}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def test_batch_inference(qwen: Qwen3VLClient):
    """Test batch processing"""
    print("\n" + "=" * 80)
    print("5. Testing Batch Inference")
    print("=" * 80)
    
    # Multiple inputs
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"},
                    {"type": "text", "text": "What type of document is this?"}
                ]
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4"},
                    {"type": "text", "text": "How long is this video?"}
                ]
            }
        ]
    ]
    
    print("\nProcessing 2 inputs in batch:")
    print("  1. Image: Receipt document type")
    print("  2. Video: Duration question")
    print("\nGenerating responses...")
    
    try:
        results = qwen.batch_generate(
            messages_batch,
            temperature=0.0,
            max_tokens=256
        )
        
        print(f"\n✅ Batch Results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result[:100]}...")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "Qwen3-VL Multimodal Test Suite" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Initialize model
    qwen = test_initialization()
    if not qwen:
        print("\n❌ Cannot continue without model. Exiting.")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Image OCR", test_image_analysis(qwen)))
    results.append(("Image Q&A", test_image_question_answering(qwen)))
    results.append(("Video Analysis", test_video_analysis(qwen)))
    results.append(("Batch Inference", test_batch_inference(qwen)))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Qwen3-VL is working correctly.")
    else:
        print("\n⚠️  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
