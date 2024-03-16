import cv2 as cv
import numpy as np
import copy

class Simulator:
    def __init__(self, map_img_file) -> None:
        self.map = cv.imread(map_img_file)
        self.pixels_per_unit = 50
        self.obs_size = 100
        self.range_x = self.map.shape[1] / self.pixels_per_unit
        self.min_x = - self.range_x / 2
        self.max_x = self.range_x / 2
        self.range_y = self.map.shape[0] / self.pixels_per_unit
        self.min_y = - self.range_y / 2
        self.max_y = self.range_y / 2

    def simulate(self):
        """Simulate moving until user presses q
        """
        coords = self.random_start()
        key, _ = self.circle_true_location(coords)
        while key != ord("q"):
            key, coords = self.step(coordinates=coords)
    
    def step(self, coordinates:np.array) -> tuple[int, np.array]:
        """move one step and circle the true location

        Args:
            coordinates (np.array): current location

        Returns:

            np.array: new location
        """
        coordinates = self.move(coordinates)
        key, _ = self.circle_true_location(coordinates)
        return key, coordinates
    
    def circle_true_location(self, true_coordinates:np.array):
        if not self.valid_location_check(true_coordinates):
            raise Exception("location out of bounds")

        # Convert from units to pixels to match the image's coordinate system
        pixel_coords = self._coordinates_to_pixels(true_coordinates)

        # Draw the circle on the map
        tmp_map = copy.deepcopy(self.map)
        cv.circle(tmp_map, (pixel_coords[0], pixel_coords[1]), radius=40, color=(255, 0, 255), thickness=20)
        cv.imshow("true location", tmp_map)
        key = cv.waitKey(0)
        return key, tmp_map
    
    def move(self, prev_coordinates:np.array) -> np.array:
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
        return new_loc
    
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
        return np.array([dx, dy])

    def observe(self, coordinates:np.array):
        """observe the image at x, y

        Args:
            coordinates (float): current x, y

        Returns:
            image: an image that's self.o
        """
        if not self.valid_location_check(coordinates):
            raise Exception("coordiantes out of bounds")

        pixel_coords = self._coordinates_to_pixels(coordinates)

        # Calculate the top-left corner of the sub-image
        start_x = pixel_coords[0] - self.obs_size // 2
        start_y = pixel_coords[1] - self.obs_size // 2

        # Extract the sub-image using slicing
        sub_image = self.map[start_y:start_y + self.obs_size, start_x:start_x + self.obs_size]

        return sub_image
    
    def valid_location_check(self, coordinates:np.array) -> bool:
        """check whether the x, y location is within bounds

        Args:
            coordinates (float): current x, y

        Returns:
            bool: True if within bounds and False otherwise
        """
        x, y = coordinates[0], coordinates[1]
        if not self.min_x <= x <= self.max_x:
            print("x coordinate out of bounds")
            return False
        if not self.min_y <= y <= self.max_y:
            print("y coordinate out of bounds")
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
    simulator = Simulator(map_img_file="MarioMap.png")
    simulator.simulate()