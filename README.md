# mbot_cv
MBot Computer Vision Starter Code and Examples
======
## 1. Colored Cone Detection
   Description: Here we train a custom Yolo-V8 model for detecting colored cones (this will theoretically work with any objects you want to detect!). We provide a starter model for you, but the model could be improved by training on more images (current model is trained on ~20 images).<br />
   Needs: Raspberry Pi or Jetson, Calibrated and Functioning MBot Camera, Google Colab, Python<br />

   1. Start by using the MBot camera to collect several pictures of the cones you want to detect in the environment you wish to detect them in. This should include at least 20 pictures, but this will work better with more pictures (~50). These pictures should include different cominations of cone poses and lighting conditions. See example photos below:<br />
      <insert example pictures><br />
   2. Create the dataset using your desired annotation tool. Here, we use Roboflow, but other options will work as well, such as LabelImg, V7, and LabelMe.<br />
      a. If this is your first time using RoboFlow, create a free account<br />
      b. Select "Create New Project"<br />
      c. Enter a descriptive Project Name, eg "MBot_Cone_Detection", type "Cones" into Annotation Group, and select Object Detection for Project Type.<br />
      d. Upload all of the images taken in Step 1 under "Upload Data".<br />
      e. Under "Annotate", select "Manual Labeling". If you wish to split the work among teammates, you can invite them to the project at this stage and assign images to label to different members.<br />
      f. Once the annotation tool pops up, use the polygonal tool to outline the cones. In this tutorial, we divide the cones into classes by color, however you could assign all cones to the same class if the color does not matter. With the desired cone outlined, type "<color>_cone" in the "Label" dialogue box. Each time you create a new label class, RoboFlow will save it so you can classify other cones of that color with the same label.<br />
      ![image](https://github.com/camharris99/mbot_cv/assets/122319358/e428483d-c5ca-4472-8494-da2458040325)<br />

      g. Repeat Step f for each cone in the image, for each image in the dataset.<br />
      h. 
   
