import cv2
import numpy as np

# Lucas-Kanade Optical Flow
def LK_Optical_Flow(prev_frame, new_frame, window_size, feature_list):
    
    tau = 0.04
    
    w = int(window_size/2)
    
    # Normalize the pixel
    prev_frame = prev_frame/255
    new_frame = new_frame/255
    
    # Gradient Operator
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])
    
    # Optical Flow matrix
    flow = np.zeros((prev_frame.shape[0],prev_frame.shape[1],2))
    u = np.zeros(prev_frame.shape)
    v = np.zeros(prev_frame.shape)
    
    # Gradient over x, y, t
    fx = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_x)
    fy = cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_y)
    ft = cv2.filter2D(src=new_frame, ddepth=-1, kernel=kernel_t) - cv2.filter2D(src=prev_frame, ddepth=-1, kernel=kernel_t)
    
    
    # Compute every deteted corner
    for feature in feature_list:
            
        j, i = feature.ravel()		
        i, j = int(i), int(j) 
            
        Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            
        b = np.reshape(It, (It.shape[0],1))
        
        A = np.vstack((Ix, Iy)).T
        
        # compute velocity
        if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
            U = np.matmul(np.linalg.pinv(A), b)
                    
            # u[i,j] = U[0]
            # v[i,j] = U[1]
            flow[i,j,0] = U[0]
            flow[i,j,1] = U[0]

    return flow

def uv_to_flow(prev_frame, feature_prev, feature):
    # Optical Flow matrix
    flow = np.zeros((prev_frame.shape[0],prev_frame.shape[1],2))
    # u = np.zeros(prev_frame.shape)
    # v = np.zeros(prev_frame.shape)
    
    i1 = feature_prev[:,1].astype(int)
    j1 = feature_prev[:,0].astype(int)

    i2 = feature[:,1]
    j2 = feature[:,0]

    flow[i1,j1,0] = i2-i1
    flow[i1,j1,1] = j2-j1
    
    return flow

def draw_optical_flow(frame, u, v, scale_factor=10):
    
    # arrow color
    line_color = (0, 0, 255)
    
    arrow_img = frame.copy()

    row, col, _  = frame.shape

    # Plot the optical flow
    for i in range(row):
        for j in range(col):
            
            if u[i,j] and v[i,j]:
                arrow_img = cv2.arrowedLine(arrow_img, (j, i), (int(round(j+v[i,j]*scale_factor)), int(round(i+u[i,j]*scale_factor))), line_color, thickness=1)
    
    return arrow_img

def remove_large_outliers(feature_list_prev, feature_list_new, threshold=97.5):
    # keep lower 97.5 percentile data
    feature_list_prev = feature_list_prev.reshape(-1,2)
    feature_list_new = feature_list_new.reshape(-1,2)

    mag = np.linalg.norm(feature_list_new - feature_list_prev, axis=1)
    ind_keep = np.argwhere(mag<=np.percentile(mag, threshold))

    feature_list_prev = feature_list_prev[ind_keep,:].reshape((ind_keep.size,2))
    feature_list_new = feature_list_new[ind_keep,:].reshape((ind_keep.size,2))

    return feature_list_prev, feature_list_new

def reshape_feature(feature_list_prev, feature_list_new):
    feature_list_prev = feature_list_prev.reshape(-1,2)
    feature_list_new = feature_list_new.reshape(-1,2)
    return feature_list_prev, feature_list_new

def show_optical_flow_hsv():
    return 

def plot_result_hsv(frame, flow, mag_max=20):
    hsv = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:,:,0] = ang * 180 / np.pi / 2
    hsv[:,:,1] = 255
    hsv[:,:,2] = (np.minimum(mag,mag_max)/mag_max*255).astype(np.uint8) # cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out

def compute_hemi_optflow_mag(flow,ds_factor=2,camera_center=413):
    mag = np.linalg.norm(flow, axis=2)
    mid = int(camera_center/ds_factor)
    optflow_mag_l = np.sum(mag[:,:mid])/mag.size
    optflow_mag_r = np.sum(mag[:,mid:])/mag.size
    return optflow_mag_l, optflow_mag_r

def calculate_ttc_v1(self,flow,foe,BLOCK_NUM_X=8,BLOCK_NUM_Y=6):
    h,w = flow.shape[:2]
    ttc = np.zeros(flow.shape[:2])
    ttc_mean = np.zeros((BLOCK_NUM_X,BLOCK_NUM_Y))

    foe_x, foe_y = foe

    i,j = np.meshgrid(np.arange(0,h),np.arange(0,w))
    ttc = np.sqrt((np.square(j - foe_x) + np.square(i - foe_y)) / (np.square(flow[:,:,0]) + np.square(flow[:,:,1])))

    rng_row = np.linspace(0,h-1,BLOCK_NUM_X+1,endpoint=True).astype(int)
    rng_col = np.linspace(0,w-1,BLOCK_NUM_Y+1,endpoint=True).astype(int)
    for i1 in range(rng_row.size-1):
        for j1 in range(rng_col.size-1):
            ttc_mean[i1,j1] = np.nanmean(ttc[rng_row[i1]:rng_row[i1+1], rng_col[j1]:rng_col[j1+1]])
    
    return ttc, ttc_mean

def calculate_FOE_v1(flow):
    # Compute the FOE
    h, w = flow.shape[:2]
    
    # horizontal foe
    h_left = np.zeros((w,))
    h_right = np.zeros((w,))

    for i in range(w):
        neg_sum = (flow[:, i, 0] < 0).sum()
        if i==0:
            h_left[i] = neg_sum
        else:
            h_left[i] = h_left[i-1] + neg_sum

    for i in np.arange(w-1,-1,-1):
        pos_sum = (flow[:, i, 0] > 0).sum()
        if i==w-1:
            h_right[i] = pos_sum
        else:
            h_right[i] = h_right[i+1] + pos_sum
    
    min_diff = float('inf')
    for x in range(w):
        diff = abs(h_left[x] - h_right[x])
        if diff < min_diff:
            min_diff = diff
            foe_x = x

    # vertical foe
    v_top = np.zeros((h,))
    v_bottom = np.zeros((h,))

    for i in range(h):
        neg_sum = (flow[i, :, 1] < 0).sum()
        if i==0:
            v_top[i] = neg_sum
        else:
            v_top[i] = v_top[i-1] + neg_sum

    for i in np.arange(h-1,-1,-1):
        pos_sum = (flow[i, :, 1] > 0).sum()
        if i==h-1:
            v_bottom[i] = pos_sum
        else:
            v_bottom[i] = v_bottom[i+1] + pos_sum

    min_diff = float('inf')
    for y in range(h):
        diff = abs(v_top[y] - v_bottom[y])
        if diff < min_diff:
            min_diff = diff
            foe_y = y

    foe = (foe_x, foe_y)
    return foe

def calculate_FOE_v2(self, flow):
    h, w = flow.shape[:2]
    h_mid, w_mid = h // 2, w // 2
    h_left = flow[:, :w_mid, 0].sum()
    h_right = flow[:, w_mid:, 0].sum()
    v_top = flow[:h_mid, :, 1].sum()
    v_bottom = flow[h_mid:, :, 1].sum()
    if abs(h_left - h_right) > abs(v_top - v_bottom):
        foe = (w_mid, h_mid)
    else:
        foe = (int(w // 2 - v_top // h_left), int(h // 2 - h_left // v_top))
    
    return foe

def calculate_ttc_v2(prev_frame, new_frame, K_inv, low_bound=0, high_bound=100):
    
    row, col = prev_frame.shape
    
    fx = cv2.Sobel(prev_frame,cv2.CV_32F,1,0,ksize=5)
    fy = cv2.Sobel(prev_frame,cv2.CV_32F,0,1,ksize=5)
    ft = new_frame.astype(np.float32) - prev_frame.astype(np.float32)
    
    x_pix,y_pix = np.meshgrid(np.arange(0,col),np.arange(0,row),indexing='xy')
    c = np.vstack([x_pix.reshape((1,-1)),y_pix.reshape((1,-1)),np.ones((1,x_pix.size))])
    
    # convert to image coordinate from pixel coordinate
    c_hat = np.dot(K_inv, c)

    # reshape back x and y
    x = c_hat[0,:].reshape(x_pix.shape)
    y = c_hat[1,:].reshape(y_pix.shape)

    # calculate radial gradient
    radial_gradient = x*fx + y*fy

    C = -np.sum(radial_gradient*ft) / np.sum(radial_gradient*radial_gradient)

    TTC = 1/C
    if TTC > high_bound or TTC < low_bound:
        TTC = np.NAN
    return TTC, C