import open3d as o3d

if __name__ == "__main__":
    o3d.visualization.webrtc_server.enable_webrtc()

    # Read point cloud
    fragment = o3d.io.read_point_cloud(
        "./mydataset/ycb_video_obj_ply/002_master_chef_can.ply"
    )

    o3d.visualization.draw_geometries(
        [fragment],
        zoom=0.7,
        front=[0.5439, -0.2333, -0.8060],
        lookat=[2.4615, 2.1331, 1.338],
        up=[-0.1781, -0.9708, 0.1608],
    )
