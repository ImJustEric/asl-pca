import os
from PIL import Image


# Convert the image to grayscale
def convert_to_grayscale(img):
	if isinstance(img, str):
		img = Image.open(img)
	return img.convert("L")


if __name__ == "__main__":
	input_root = "data/raw"
	output_root = "data/filtered"
	image_exts = {".jpg", ".jpeg", ".png"}

    # Go through each folder and preocess each image
	for root, _, files in os.walk(input_root):
		for filename in files:
			ext = os.path.splitext(filename)[1].lower()
			if ext not in image_exts:
				continue

			input_path = os.path.join(root, filename)
			relative_path = os.path.relpath(input_path, input_root)
			output_path = os.path.join(output_root, relative_path)
			
            # Change to _gray as end of file
			stem = os.path.splitext(os.path.basename(output_path))[0]
			if stem.endswith("_cropped"):
				stem = stem[: -len("_cropped")] + "_gray"
			output_path = os.path.join(os.path.dirname(output_path), f"{stem}{ext}")

			os.makedirs(os.path.dirname(output_path), exist_ok=True)
			gray = convert_to_grayscale(input_path)
			gray.save(output_path)

