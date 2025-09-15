import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

dataset = fetch_olivetti_faces(data_home="./") #현재 디렉토리에 데이터 셋 저장
data = dataset.data
print(data.shape) #400images and 4096 pixels

fig = plt.figure(figsize=(12, 12))
plt.imshow(np.concatenate(dataset.data.reshape((-1,64,64))[:400:40], axis=1), cmap="gray")
#400장의 사진이 64*64로 되며 이것을 0부터 399까지 40씩 건너뛰며 10장의 사진 axis = 1 -> 가로 추출
##Important information:::: The 400 images represent -> samge person per 10 images
plt.show()

mean_face = np.mean(data, axis=0)
shift_data = data - mean_face

def compute_covariance(shift_data): #covariance를 계산하기 위한 함수
    n = shift_data.shape[0]
    covarianve = (shift_data.T @ shift_data)/(n-1)

    return covarianve

def comput_pca(cov): #앞서 구한 공분산으로 고윳값과 고유백터를 구하는 함수
    eigenvalues, eigenvectors = np.linalg.eigh(cov) #get eignevalues and eigenvectors using function

    sorted_indices = np.argsort(eigenvalues) [::-1]
    eigenvalues = eigenvalues[sorted_indices] #standard: column
    eigenvectors = eigenvectors[:, sorted_indices] #standard: row

    return eigenvalues, eigenvectors

eigenvalues, eigenvectors = comput_pca(compute_covariance(shift_data))

eigenvalues = eigenvalues.real #실수만 취급
eigenvectors = eigenvectors.real

def explained_variance(eigenvalues):
    total_var = np.sum(eigenvalues)
    explained_var = (eigenvalues / total_var) * 100
    return explained_var

#Percentage of 5 variances being explained by each principal component 
print(explained_variance(eigenvalues)[:5]) #절대 분산: pca앞의 5개를 나타냄. pca개념에 따라 첫번째가 가장 큰 분산을 나타냄
fig = plt.figure(figure=(12, 6))

explained_var = explained_variance(eigenvalues)
cumulative_exp_var = np.cumsum(explained_var) #It means that how much data can be represented as Percentage!
plt.plot(explained_var[:100], label="explained variance per component", marker="o")
plt.plot(cumulative_exp_var[:100], label="cumulative explained variance", marker="o")
plt.xlabel("components")
plt.ylabel("Percentage of variance")
plt.ylim(0, 100)
plt.legend()
plt.show()

def project_to_principal_components(shifted_data, pca): #Using this function, we can represent the oreiginal data to reduced dimensional space
    project = shifted_data @ pca #원래의 데이터와 고유벡터를 곱하여 낮은 차원에서의 데이터를 나타낼 수 있음
    return project

projected_data = project_to_principal_components(shift_data, eigenvectors[:,:2]) #TWO dimensional space

plt.figure(figsize=(10, 10))
x_points = projected_data[:,0]#first principal com
y_points = projected_data[:,1]#second principal com
color = [f"C{i}" for i in range(10)]
marker_styles = ["o", "x", "v", "*", "8", "P", "D", "p", "s", "X", "."]
for i in range(len(np.unique(dataset.target))): #dataset.target에는 40개의 라벨이 있음. So 40 people in 400 imagesd
    plt.scatter(
        x_points[dataset.target == i],
        y_points[dataset.target == i],
        color = color[i % 10],
        marker=marker_styles[i // 10],
        s = 100
    )

plt.xlabel('x')
plt.ylabel('y')
plt.show()

def reconstruct_from_principal_components(projected_data, eigencvectors, data_mean): #Using this fuction, we can get original(but a bit modified) data
    reconstruct = projected_data @ eigencvectors.T
    reconstruct += data_mean #Because we used shifted data
    return reconstruct

max_num_components = 120 #120-dimension among thtoly 4096(64*64) dimension
face_id = 5
projected_data = project_to_principal_components(shift_data, eigenvectors[:, :max_num_components]) #All 40 people -> 120dimension
reconstructions = []

for i in range(max_num_components): #1-dimension to 120 dimension
    reconstructed_data = reconstruct_from_principal_components(projected_data[face_id, :i], eigenvectors[:, :i], mean_face) #Until the index i
    reconstructions.append(reconstructed_data) 

def plot_faces(faces, stride=1):
    faces = faces[::stride] #array[start:stop:step]
    faces = np.array(faces)
    plt.imshow(np.concatenate(faces.reshape(-1, 64, 64), axis = 1), cmap = "gray") #64*64- images, and perpendicular(가로)
    #-1:행 개수를 자동 계산
    plt.axvline((len(faces)-1)*64 - 0.5, color="red") 
    plt.xticks(np.arange(len(faces))*64 -32, (np.arange(len(faces))-1)*stride) #페이스의 갯수*64(가로 픽셀 수)를 중앙(32)으로, 사용 주성분 갯수
    plt.xlim(0, len(faces)*64)
    plt.xlabel("principal components")
    plt.show() #in other words, x-axis: the number of principal com. and 0부터 64까지의 각 이미지, 

plot_faces(reconstructions[:12], stride=1) #상위 1-12개의 주성분

plot_faces(reconstructions, stride=10) #총 설정한 주성분 중 10개 단위로