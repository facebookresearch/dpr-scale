import numpy as np
import faiss

class ProductQuantizer:
    def __init__(self, dim=32, sub_vec_dim=4, num_centroids=256, iter=5, mode="train"):
        self.dim = dim
        self.num_sub_vecs = dim // sub_vec_dim
        self.sub_vec_dim = sub_vec_dim
        self.num_centroids = num_centroids
        self.quantizer = faiss.ProductQuantizer(dim, self.num_sub_vecs, int(np.log2(num_centroids)))
        if mode == "train":
            self.kmeans = faiss.Kmeans(sub_vec_dim, num_centroids, niter=iter, nredo=1, gpu=1)
    
    def fit(self, vecs):
        codewords = np.zeros((self.num_sub_vecs, self.num_centroids, self.sub_vec_dim), dtype=np.float32)
        for m in range(self.num_sub_vecs):
            vecs_sub = vecs[:, m * self.sub_vec_dim : (m + 1) * self.sub_vec_dim]
            self.kmeans.train(np.ascontiguousarray(vecs_sub))
            codewords[m] = self.kmeans.centroids
        self.set_centroids(codewords)    

    def encode(self, vecs):
        codes = self.quantizer.compute_codes(vecs)
        return codes
    
    def decode(self, codes):
        vecs = self.quantizer.decode(codes)
        return vecs

    def get_centroids(self):
        centroids = faiss.vector_to_array(self.quantizer.centroids).reshape(self.num_sub_vecs, self.num_centroids, self.sub_vec_dim)
        return centroids

    def set_centroids(self, centroids):
        faiss.copy_array_to_vector(centroids.ravel(), self.quantizer.centroids)
