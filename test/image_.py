from PIL import Image, ImageDraw, ImageFont

# Load the image
input_path = "/mnt/data/3MLP.png"
output_path = "/mnt/data/annotated_neural_network.png"
img = Image.open(input_path)

# Define font and size
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Create draw object
draw = ImageDraw.Draw(img)

# Define neuron count and positions to annotate
neuron_count = "64"
positions = [(215, 320), (360, 320), (505, 320), (650, 320)]

# Annotate the image
for pos in positions:
    draw.text(pos, neuron_count, fill="black", font=font)

# Save the annotated image
img.save(output_path)
img.show()
