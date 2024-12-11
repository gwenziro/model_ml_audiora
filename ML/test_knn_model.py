from model_testing import predict_age
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_knn_model.py <image_path>", file=sys.stderr)
        sys.exit(1)

    test_image_path = sys.argv[1]
    predicted_label = predict_age(test_image_path)
    print(predicted_label)