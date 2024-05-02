# WIP limitations/thoughts
- dust3r resizes images
- scene_radius only applies for inward facing cameras, got to think abt workaround
- varying cam intrinsics even if we know that they are all the same, proxy to dust3r being a little sussy
- holy cannoli that's a lot of hyper-params, trying to put them all into hydra config files, maybe run some experiments
- instead of directly using the sfm pt cloud to init gaussians, we could use them to generate a space probability distribution of gaussians, then sample from that. *this is mega stretch*
- new idea(?) train a gs on low res images, then render on high res. maybe can be done with 2d gaussians/gaussian surfels?