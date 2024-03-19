import cv2 as cv
import numpy as np
import torch
import random
from pytorch_msssim import ssim
from sklearn.cluster import spectral_clustering


class ParticleFilter:
    def __init__(self, N, min_x, max_x, min_y, max_y, device="cpu") -> None:
        self.num_particles = N
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.particles = self._initialize_particles()
        self.weights = torch.tensor([1.0/self.num_particles]*self.num_particles)
        self.device = device


    
    def update_weights(self, imgs:np.array, ref_img:np.array, gps_reading:np.array, dist_sim_weight=0.8):
        """filter the particles

        Args:
            imgs (np.array): img observed at each particle
            ref_img (np.array): image observed at true location
            gps_reading (np.array): gps reading of the true location
        """
        if len(gps_reading.shape) > 0:
            sim_scores = self.prob_obs_given_coords_and_gps(imgs, ref_img, gps_reading, dist_sim_weight=dist_sim_weight)
        else:
            sim_scores = self.prob_obs_given_coords(imgs=imgs, ref_img=ref_img)
        weights = self.assign_weight(sim_scores=sim_scores)
        self.weights = weights
        return weights


    def move(self, move_vector:np.array):
        """move the particles according to a movement vector [dx, dy] that the simulator generates

        Args:
            move_vector (np.array): [dx, dy] movement vector
        """
        moved_particles = self.particles + move_vector
        noise = np.random.normal(0, 0.1, size=(len(moved_particles), 2))
        # add a little noise per particle
        moved_particles = moved_particles + noise
        rand_particles = self._initialize_particles()
        # replace out of bounds particles with new random ones
        out_of_bounds = np.where(
            (moved_particles[:, 0] < self.min_x) | (moved_particles[:, 0] > self.max_x) |
            (moved_particles[:, 1] < self.min_y) | (moved_particles[:, 1] > self.max_y)
        )[0]
        moved_particles[out_of_bounds] = rand_particles[out_of_bounds]
        self.particles = moved_particles

    # EXTRA CREDIT 1: Image Comparison
    # EXTRA CREDIT 2: GPS readings
    def prob_obs_given_coords_and_gps(self, imgs, ref_img:np.array, gps_reading:np.array, dist_sim_weight=0.8) -> torch.Tensor:
        """Calculate the probabilities of observing the input images given current state a.k.a a pair of coordinates

        Args:
            imgs (MatLike): observed images at different particles
            ref_img (MatLike): reference image
            gps_reading (np.array): gps reading of the true location with some noise
            dist_sim_weight (float): how much weight is given to the similarity based on distance in the composite score
        Returns:
            float: p(z|state)
        """

        # error based on gps reading
        norm_denom = np.sqrt((self.max_x - self.min_x)**2 + (self.max_y - self.min_y)**2)
        error_from_gps = np.sqrt(np.sum((self.particles - gps_reading)**2, axis=1))
        # error in the range [0.0, 1.0]
        error_from_gps_normalized = error_from_gps / norm_denom
        # flip error into similarity
        dist_sim_scores = 1.0 - error_from_gps_normalized

        # turn both into tensors
        # (B, H, W, C)
        imgs_list = []
        for img in imgs:
            img = self._convert_to_tensor(img, to_grey=False)
            imgs_list.append(img)
        img_batch = torch.stack(imgs_list)

        # make the reference image into a batch of the same shape
        ref_img = self._convert_to_tensor(ref_img, to_grey=False)
        ref_img_batch = ref_img.unsqueeze(0).repeat(len(imgs_list), 1, 1, 1)
        img_batch = img_batch.to(self.device)
        ref_img_batch = ref_img_batch.to(self.device)
        # compare the batch of images to the reference image
        ssim_scores = ssim(img_batch, ref_img_batch, data_range=1.0, size_average=False)
        
        # assumes the particles are in the same order as the imgs, combine the two measures
        combined_measure = (1.0 - dist_sim_weight) * ssim_scores + dist_sim_weight * dist_sim_scores

        return combined_measure
    
    def prob_obs_given_coords(self, imgs, ref_img:np.array) -> torch.Tensor:
        """Calculate the probabilities of observing the input images given current state a.k.a a pair of coordinates

        Args:
            imgs (MatLike): observed images at different particles
            ref_img (MatLike): reference image
        Returns:
            float: p(z|state)
        """

        # turn both into tensors
        imgs_list = []
        for img in imgs:
            img = self._convert_to_tensor(img, to_grey=False)
            imgs_list.append(img)
        img_batch = torch.stack(imgs_list)

        # make the reference image into a batch of the same shape
        ref_img = self._convert_to_tensor(ref_img, to_grey=False)
        ref_img_batch = ref_img.unsqueeze(0).repeat(len(imgs_list), 1, 1, 1)
        img_batch = img_batch.to(self.device)
        ref_img_batch = ref_img_batch.to(self.device)
        # compare the batch of images to the reference image
        ssim_scores = ssim(img_batch, ref_img_batch, data_range=1.0, size_average=False)
        #ssim_scores = (ssim_scores + 1.0) / 2.0
        #ssim_scores = torch.sigmoid(ssim_scores)
        return ssim_scores
    
    
    def resample(self, weights:torch.Tensor):
        """Resample particles based on their similarity scores

        Args:
            weights (torch.Tensor): list of weights

        Returns:
            torch.Tensor: list of resampled particles
        """
        new_particles = self._roulette_resample(weights=weights, percent=0.95)
        new_rand_particles = self._initialize_particles()[:int(len(self.particles)*0.05)]
        # add some number of random particles
        self.particles = np.concatenate((new_particles, new_rand_particles), axis=0)[:self.num_particles]
        # reset weights
        self.weights = torch.tensor([1.0/self.num_particles]*self.num_particles)

    
    def _roulette_resample(self, weights:torch.Tensor, percent=0.95) -> np.array:
        """Resample particles based on their similarity scores to the reference image

        Args:
            sim_scores (torch.Tensor): list of scores
            percent (float): what percentage of the particles to resample using this method

        Returns:
            torch.Tensor: list of resampled particles
        """
        new_particles = []
        while len(new_particles) < int(self.num_particles*percent):
            arr_cusum = np.cumsum(weights)
            rand_float = np.random.uniform(0, arr_cusum[-1])
            index = np.where(arr_cusum >= rand_float)[0][0]
            new_particles.append(self.particles[index])
        return np.array(new_particles)
    
    def assign_weight(self, sim_scores:torch.Tensor, normalize=False) -> torch.Tensor:
        """Assign weights to particles based on their p(z|x) scores

        Args:
            sim_scores (torch.Tensor): similarity scores between the observations at the particles and the true observation

        Returns:
            torch.Tensor: a list of weights, one weight for each particle
        """
        if normalize:
            return torch.true_divide(sim_scores, torch.sum(sim_scores))
        else:
            return sim_scores

    
    def _convert_to_tensor(self, img, to_grey=False) -> torch.Tensor:
        """Convert opencv image to tensor"""
        if to_grey:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = img.astype("float32") / 255.0
            # make shape (C, H, W) instead of (H, W, C)
            tensor = torch.from_numpy(img).unsqueeze(0)
        else:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0
            # make shape (C, H, W) instead of (H, W, C)
            tensor = torch.from_numpy(img).permute(2, 0, 1)
        # Ensure image is of type uint8
        # if img.dtype != np.uint8:
        return tensor
    
    def _initialize_particles(self) -> np.array:
        """Initialize particples such that they are uniformly distributed on the map

        Returns:
            np.array: an array of particle (x, y) locations
        """
        coords = np.random.uniform(low=(self.min_x, self.min_y), high=(self.max_x, self.max_y), size=(self.num_particles, 2))
        return coords

    
if __name__=="__main__":
    filter = ParticleFilter(N=100)
    particle_imgs = [filter.simulator.observe(particle_coords) for particle_coords in filter.particles]
    # true location (0, 0)
    ref_img = filter.simulator.observe(np.array([0,0]))
    probs = filter.prob_obs_given_coords(particle_imgs, np.array([0,0]))
    closest = torch.argmax(probs)
    furthest = torch.argmin(probs)
    cv.imshow("ref", ref_img)
    cv.waitKey(0)
    cv.imshow("closest", particle_imgs[closest])
    cv.waitKey(0)
    cv.imshow("furthest", particle_imgs[furthest])
    cv.waitKey(0)
    print(probs)