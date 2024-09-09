import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class Particle:
    def __init__(self, image_shape):
        self.position = np.random.uniform(0, 255, image_shape)
        self.velocity = np.random.uniform(-10, 10, image_shape)
        self.best_position = self.position.copy()
        self.best_fitness = -float('inf')  # Initial fitness set to negative infinity


def objective_function(image, reference):
    # Calculate SSIM between the image and reference image
    score, _ = ssim(image, reference, full=True)

    return score


def pso(image, reference, num_particles, max_iter, inertia_weight=0.5, cognitive_weight=1.5, social_weight=1.5):
    # Initialize swarm
    image_shape = image.shape
    swarm = [Particle(image_shape) for _ in range(num_particles)]

    # Initialize global best
    global_best_position = None
    global_best_fitness = -float('inf')

    # Iterate over generations
    for _ in range(max_iter):
        for particle in swarm:
            # Evaluate fitness
            fitness = objective_function(particle.position, reference)

            # Update personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

            # Update velocity
            particle.velocity = (inertia_weight * particle.velocity +
                                 cognitive_weight * np.random.rand(*image_shape) * (
                                             particle.best_position - particle.position) +
                                 social_weight * np.random.rand(*image_shape) * (
                                             global_best_position - particle.position))

            # Update position
            particle.position += particle.velocity

            # Clip position to ensure within bounds (0-255)
            particle.position = np.clip(particle.position, 0, 255)

    return global_best_position, global_best_fitness


# Load reference and distorted images with raw string literals
reference_image = cv2.imread(r"/content/bridge.png", cv2.IMREAD_GRAYSCALE)
distorted_image = cv2.imread(r"/content/bridge.JPEG.2.png", cv2.IMREAD_GRAYSCALE)

# Check if images were loaded successfully
if reference_image is None:
    print("Error: Failed to load reference image")
    exit()

if distorted_image is None:
    print("Error: Failed to load distorted image")
    exit()

# Set PSO parameters
num_particles = 50
max_iter = 100

# Perform PSO optimization
best_image, best_fitness = pso(distorted_image, reference_image, num_particles, max_iter)

# Output results
print("Best Fitness (SSIM):", best_fitness)
print("SSIM:", score)

# Optionally, save or display the best optimized image
if best_image is not None:
    cv2.imwrite("best_image.jpg", best_image)
else:
    print("Error: Failed to generate best image")
