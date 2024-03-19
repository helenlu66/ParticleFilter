import cv2 as cv
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import os
from pathlib import Path
from ParticleFilter import ParticleFilter

class Simulator:
    def __init__(self, map_img_file, N=1000, obs_size=100, dist_sim_weight=0.8, K=100, device='cpu', error_thresh=0.05, white_noise=0.0) -> None:
        self.map = cv.imread(map_img_file)
        self.file = map_img_file
        self.pixels_per_unit = 50
        self.obs_size = obs_size
        self.padded_map = cv.copyMakeBorder(self.map, self.obs_size, self.obs_size, self.obs_size, self.obs_size, cv.BORDER_CONSTANT, value=[0, 0, 0])
        self.range_x = self.map.shape[1] / self.pixels_per_unit
        self.min_x = - self.range_x / 2
        self.max_x = self.range_x / 2 - 1
        self.range_y = self.map.shape[0] / self.pixels_per_unit
        self.min_y = - self.range_y / 2
        self.max_y = self.range_y / 2 - 1
        self.filter = ParticleFilter(N=N, min_x=self.min_x, max_x=self.max_x, min_y=self.min_y, max_y=self.max_y, device=device)
        self.K = K
        self.dist_sim_weight = dist_sim_weight
        self.dir = self.file.split(".")[0]
        self.error_thresh = error_thresh
        self.white_noise = white_noise

    def simulate(self, show_img=False):
        """Simulate moving until user presses q
        Args:
            particles (np.array): particles to be drawn on the map
        """
        # save images in this directory
        dir = Path(f"{self.dir}/obs_size{self.obs_size}/pnum{self.filter.num_particles}/score_weight{self.dist_sim_weight}/noise{self.white_noise}")
        dir.mkdir(parents=True, exist_ok=True)
        # document the change in error
        errors = []
        coords = self.random_start()
        _ = self.circle_particles()
        _ = self.circle_location(coords, radius=int(self.range_x*self.pixels_per_unit//50), thickness=int(max(3, self.range_x//10)))
        i = 0
        if show_img:
            cv.imshow(f"step {i}", self.map)
            key = cv.waitKey(0)
        while i < self.K:
            # reset the map
            self.map = cv.imread(self.file)
            coords, _ = self.step(coordinates=coords)
            # get a gps reading
            reading = self.gps(coords=coords)
            #print(f"true coords: {coords}   reading: {reading}")
            # observe image at each particle
            imgs = [self.observe(p) for p in self.filter.particles]
            self.filter.update_weights(imgs=imgs, ref_img=self.add_white_noise(self.observe(coords), level=self.white_noise), gps_reading=reading, dist_sim_weight=self.dist_sim_weight)
            error = self.eval(coordinates=coords)
            errors.append(error)
            # if error <= self.error_thresh:
            #     break
            # draw circles for particles with updated weights
            self.circle_particles()
            self.circle_location(coords, radius=int(self.range_x*self.pixels_per_unit//50), thickness=int(max(3, self.range_x//10)))
            if show_img:
                cv.imshow(f"step {i}: weights", self.map)
                key = cv.waitKey(0)
                if key == ord("q"):
                    break
            if i % 10 == 0:cv.imwrite(str(dir/f"step{i}weights.png"), self.map)
            # resample particles
            self.map = cv.imread(self.file)
            self.filter.resample(weights=self.filter.weights)
            self.circle_particles()
            self.circle_location(coords, radius=int(self.range_x*self.pixels_per_unit//50), thickness=int(max(3, self.range_x//4)))
            if i % 10 == 0:cv.imwrite(str(dir/f"step{i}resampled.png"), self.map)
            if show_img:
                cv.imshow(f"step {i}: resampled particles", self.map)
                key = cv.waitKey(0)
                if key == ord("q"):
                    break
            
            i += 1
            
        return errors
            
    
    def eval(self, coordinates:np.array):
        """Evaluate the fitness of the current particles

        Args:
            coordiantes (np.array): current true location
        """
        # find the top 10% highest similarity indices, essentially kNN
        k = int(self.filter.num_particles * 0.1)
        _, indices = torch.topk(self.filter.weights, k)
        # get the top 10% most similar particles
        topk_particles = torch.tensor(self.filter.particles[indices])
        true_location = torch.tensor(coordinates).expand_as(topk_particles)
        criterion = nn.MSELoss()
        mse = criterion(topk_particles.float(), true_location.float())
        return mse.item()
    
    # EXTRA CREDIT 3: adding noise to reference image
    def add_white_noise(self, img, level):
        """Add Gaussian noise to image

        Args:
            img (MatLike): image
            level (float): noise level
        """
        h, w, c = img.shape
        if level > 0.0:
            noise = np.random.normal(0, level, size=(h, w, c))*255
            noisy_img = cv.add(img, noise.astype(np.uint8))
            return noisy_img
        else:
            return img

    
    def gps(self, coords:np.array, scale=1.0):
        """generate a noisy gps reading

        Args:
            coords (np.array): true location
        """
        new_loc = np.array([float("inf"), float("inf")])
        while not self.valid_location_check(new_loc):
            noise = np.random.normal(loc=0.0, scale=scale, size=(2,))
            new_loc = coords + noise
        return new_loc
        
    
    def inspect(self, imgs):
        for i, img in enumerate(imgs):
            cv.imshow(f"particle{i}", img)
            cv.waitKey(0)

    
    def step(self, coordinates:np.array) -> tuple[int, np.array, np.array]:
        """move one step and circle the true location and particles

        Args:
            coordinates (np.array): current location

        Returns:

            np.array: new location
        """
        coordinates, move_vector = self.move(coordinates)
        self.filter.move(move_vector=move_vector)
        return coordinates, move_vector
    
    def circle_particles(self):
        """draw a circle for each particle
        """
        for p, weight in zip(self.filter.particles, self.filter.weights):
            self.circle_location(coords=p, radius=int(max(1, (self.range_x*self.pixels_per_unit*weight//50))), color=(180, 105, 255), thickness=int(min(5, self.range_x//4)))
        return self.map

    
    def circle_location(self, coords:np.array, radius=20, color=(255, 255, 0), thickness=3):
        if not self.valid_location_check(coords):
            raise Exception("location out of bounds")

        # Convert from units to pixels to match the image's coordinate system
        pixel_coords = self._coordinates_to_pixels(coords)

        # Draw the circle on the map
        cv.circle(self.map, (pixel_coords[0], pixel_coords[1]), radius=radius, color=color, thickness=thickness)
        return self.map
    
    
    def move(self, prev_coordinates:np.array) -> tuple[np.array, np.array]:
        """move for one time step

        Args:
            prev_coordinate (np.array): x, y coordinates

        Returns:
            np.array: the current x, y coordinates after moving
        """
        new_loc = np.array([float("inf"), float("inf")])
        while not self.valid_location_check(new_loc):
            rand_move_vecor = self.random_move_vector()
            noise = np.random.normal(0, 0.1, size=(2,))
            new_loc = prev_coordinates + rand_move_vecor + noise
        return new_loc, rand_move_vecor
    
    def random_start(self):
        """Generate a random start (x, y)
        """
        x = np.random.uniform(low=self.min_x, high=self.max_x)
        y = np.random.uniform(low=self.min_y, high=self.max_y)
        return np.array([x, y])

    def random_move_vector(self) -> np.array:
        """Generate a random movement vector

        Args:
            coordinates (float): current x, y
        Returns:
            np.array: 2D movement vector
        """
        
        dx = np.random.uniform(low=-1.0, high=1.0)
        dy = np.sqrt(1.0 - dx**2)
        choices = np.array([np.array([dx, dy]), np.array([dx, -dy])])
        return random.choice(choices)

    def observe(self, coordinates:np.array):
        """observe the image at x, y

        Args:
            coordinates (float): current x, y

        Returns:
            image: an image that is a section of the map
        """
        if not self.valid_location_check(coordinates):
            raise Exception("coordiantes out of bounds")

        pixel_coords = self._coordinates_to_pixels(coordinates)

        # Calculate the top-left corner of the sub-image
        start_x = int(pixel_coords[0] - self.obs_size // 2 - ( - self.obs_size))
        start_y = int(pixel_coords[1] - self.obs_size // 2 - ( - self.obs_size))

        # Extract the sub-image using slicing
        sub_image = self.padded_map[start_y:start_y + self.obs_size, start_x:start_x + self.obs_size]

        return sub_image
    
    def valid_location_check(self, coordinates:np.array) -> bool:
        """check whether the x, y location is within bounds

        Args:
            coordinates (np.array): current x, y

        Returns:
            bool: True if within bounds and False otherwise
        """
        x, y = coordinates[0], coordinates[1]
        if not self.min_x <= x <= self.max_x:
            #print("x coordinate out of bounds")
            return False
        if not self.min_y <= y <= self.max_y:
            #print("y coordinate out of bounds")
            return False
        return True
    
    def _coordinates_to_pixels(self, coordinates:np.array) -> np.array:
        """convert the given coordinates to pixels

        Args:
            coordinates (np.array): x, y

        Returns:
            np.array: pixel x, y
        """
        # Convert from units to pixels to match the image's coordinate system
        pixel_x = int((coordinates[0] - self.min_x) * self.pixels_per_unit)
        pixel_y = int((coordinates[1] - self.min_y) * self.pixels_per_unit)
        return np.array([pixel_x, pixel_y])


if __name__=="__main__":
    # test if white noise is working
    simulator = Simulator(map_img_file="BayMap.png", N=1000, K=60, obs_size=500, white_noise=0.0)
    img = simulator.observe((0,0))
    noisy_img = simulator.add_white_noise(img, level=0.0)
    cv.imshow('no noise', noisy_img)
    cv.waitKey(0)
    noisy_img = simulator.add_white_noise(img, level=0.25)
    cv.imshow('25', noisy_img)
    cv.waitKey(0)
    noisy_img = simulator.add_white_noise(img, level=0.5)
    cv.imshow('50', noisy_img)
    cv.waitKey(0)
    noisy_img = simulator.add_white_noise(img, level=0.75)
    cv.imshow('75', noisy_img)
    cv.waitKey(0)
    noisy_img = simulator.add_white_noise(img, level=1)
    cv.imshow('1', noisy_img)
    cv.waitKey(0)
