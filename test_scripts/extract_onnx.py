import ast
import onnx

# 1. Load the ONNX model file
model = onnx.load("/home/robosub/robosub2027/robosub-2027-ws/test_scripts/best.onnx")

# 2. Extract metadata properties into a Python dictionary
metadata_props = {p.key: p.value for p in model.metadata_props}

# 3. Look for the standard 'names' key used by popular frameworks
if "names" in metadata_props:
    # Safely convert the string representation of the dictionary/list back to Python
    class_names = ast.literal_eval(metadata_props["names"])
    print("Class names found:", class_names)
else:
    print("No class 'names' key found in this model's metadata properties.")
