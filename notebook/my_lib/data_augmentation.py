import numpy as np
import cv2

class DataAugmentation(object):
    """
    A class that generate data augmetantion
    """

    def __init__(self, images, options=None):
        options = {} if options is None else options
        self.images = images[:]
        self.length = len(images)
        
        self.horizontal_flips = options.get('horizontal_flips', False)
        self.rotation = options.get('rotation', False)
        self.rotation_config = options.get('rotation_config', [
            (10,1.2), (20,1.3)
        ])
        self.inverse = options.get('inverse', False)
        self.sobel_derivative = options.get('sobel_derivative', False)
        self.scharr_derivative = options.get('scharr_derivative', False)
        self.laplacian = options.get('laplacian', False)
        self.blur = options.get('blur', False)
        self.blur_config = options.get('blur_config', {
            'kernel_size': 15,
            'step_size': 2
        })
        self.gaussian_blur = options.get('gaussian_blur', False)
        self.gaussian_blur_config = options.get('gaussian_blur_config', {
            'kernel_size': 20,
            'step_size': 2
        })
        self.median_blur = options.get('median_blur', False)
        self.median_blur_config = options.get('median_blur_config', {
            'kernel_size': 10,
            'step_size': 2
        })
        self.bilateral_blur = options.get('bilateral_blur', False)
        self.bilateral_blur_config = options.get('bilateral_blur_config', {
            'kernel_size': 30,
            'step_size': 2
        })
        self.shuffle_result = options.get('shuffle_result', False)

    def __iter__(self):
        for image in self.images:
            augmented_image_set = []

            if self.inverse:
                augmented_image_set.append((255 - image))

            if self.sobel_derivative:
                #derivatives = []
                #for image in images:
                n_image = cv2.GaussianBlur(image, (3, 3), 0)
                gray = cv2.cvtColor(n_image, cv2.COLOR_RGB2GRAY)
                grad_x = cv2.Sobel(
                    gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
                )
                grad_y = cv2.Sobel(
                    gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
                )
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                n_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                dst = cv2.cvtColor(n_image, cv2.COLOR_GRAY2RGB)
                augmented_image_set.append(dst)
                #ugmented_images_set += derivatives

            if self.scharr_derivative:
                #derivatives = []
                #for image in images:
                n_image = cv2.GaussianBlur(image, (3, 3), 0)
                gray = cv2.cvtColor(n_image, cv2.COLOR_RGB2GRAY)
                grad_x = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
                grad_y = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                n_image = cv2.add(abs_grad_x, abs_grad_y)
                dst = cv2.cvtColor(n_image, cv2.COLOR_GRAY2RGB)                
                augmented_image_set.append(dst)
                #derivatives.append(dst)
                #augmented_images_set += derivatives

            if self.laplacian:
                # FIXME: openCV error
                # pass
                n_image = cv2.GaussianBlur(image, (3, 3), 0)
                gray = cv2.cvtColor(n_image, cv2.COLOR_RGB2GRAY)
                gray_lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3, scale=1, delta=0)
                n_image = cv2.convertScaleAbs(gray_lap)
                dst = cv2.cvtColor(n_image, cv2.COLOR_GRAY2RGB)
                augmented_image_set.append(dst)

            if self.blur:
                # augmented_images_set += np.hstack([
                # 		[
                # 			cv2.blur(image, (i, i))
                # 			for i in range(1, blur_config['kernel_size'], blur_config['step_size'])
                # 		]
                # 		for image in images
                # 	]).tolist()

                for i in range(1, self.blur_config['kernel_size'], self.blur_config['step_size']):
                    augmented_image_set.append(cv2.blur(image, (i, i)))

            if self.gaussian_blur:
                # augmented_images_set += np.hstack([
                # 		[
                # 			cv2.GaussianBlur(image, (i, i), 0)
                # 			for i in range(1, gaussian_blur_config['kernel_size'], gaussian_blur_config['step_size'])
                # 		]
                # 		for image in images
                # 	]).tolist()
                for i in range(1, self.gaussian_blur_config['kernel_size'], self.gaussian_blur_config['step_size']):
                    augmented_image_set.append(cv2.GaussianBlur(image, (i+1, i), 0))

            if self.median_blur:
                # augmented_images_set += np.hstack([
                # 		[
                # 			cv2.medianBlur(image, i)
                # 			for i in range(1, median_blur_config['kernel_size'], median_blur_config['step_size'])
                # 		]
                # 		for image in images
                # 	]).tolist()
                for i in range(1, self.median_blur_config['kernel_size'], self.median_blur_config['step_size']):
                    augmented_image_set.append(cv2.medianBlur(image, i))

            if self.bilateral_blur:
                # augmented_images_set += np.hstack([
                # 		[
                # 			cv2.bilateralFilter(image, i, i*2, i/2)
                # 			for i in range(1, bilateral_blur_config['kernel_size'], bilateral_blur_config['step_size'])
                # 		]
                # 		for image in images
                # 	]).tolist()
                for i in range(1, self.bilateral_blur_config['kernel_size'], self.bilateral_blur_config['step_size']):
                    augmented_image_set.append(cv2.bilateralFilter(image, i, i*2, i/2))


            # if not augmented_image_set:
            # augmented_image_set.append(image)

            if self.rotation:
                # for image in augmented_image_set:
                #     rows, cols, *_ = image.shape
                #     for angle, scale in [(-20,1.3), (-10,1.2), (10,1.2), (20,1.3)]:
                #         M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                #         dst = cv2.warpAffine(image, M, (cols, rows))
                #         augmented_image_set.append(dst)
                
                rotations = self.rotation_config
                def rotate_image(image, angle, scale):
                    rows, cols, *_ = image.shape                    
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
                    dst = cv2.warpAffine(image, M, (cols, rows))
                    return dst
                augmented_image_set += [rotate_image(image, angle, scale) 
                                        for image in augmented_image_set + [image]
                                        for angle, scale in rotations]
           
            if self.horizontal_flips:
                augmented_image_set += [cv2.flip(np.array(image), 1) for image in augmented_image_set + [image]]

            if self.shuffle_result:
                np.random.shuffle(augmented_image_set)
            
            yield augmented_image_set


    def __len__(self):
        """This method returns the total number of elements"""
        return self.length
