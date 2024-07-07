# Cursor-Mover

## Project Description

We aim to develop a robust voice command recognition system, capable of understanding and responding to specific spoken instructions.  
Our application aims to enable handicapped people to control the computer through AI-assisted mouse control.

## Motivation

[Provide instructions on how to get started with your project, including any necessary software or data. Include installation instructions and any prerequisites or dependencies that are required.]

## Research Method
1. Data collection
Audio clips are essential for training a voice recognition system. We collect the audio files for this model and save them on the device where we want to train the model. Below are the details of how we collect data.
 - Recording Audio Clips
To ensure the diversity of audio files, we collected data by asking people with different accents to help record the clips. By doing so, we can increase the stability in accuracy rate when under different accents and tones. We gathered 415 clips for each word, namely, 2075 files in total. All the files were stored in WAV form.
 - Labeling 
After the collection process, the files are categorized into corresponding classes: up, down, left, right and click. In each folder, clips are labeled with serial numbers (1 to 415.)
2. Coding
  - Installing and Importing Necessary Libraries
Due to the space constraint, only some special and important libraries, training processes and problem-solving methods used in the project are listed and explained below. For the complete coding, please refer to the attachment at the end of this document.
    - Installing Libraries
pydub
      - “pydub,” a Python library for audio manipulation, is needed to integrate audio processing capabilities into Python projects. It allows the performance of a variety of audio processing tasks such as cutting, concatenating, and converting audio files.
FFmpeg
      - “FFmpeg” is required by many audio and video processing libraries, including “pydub”, because it provides the necessary backend functionalities to decode, encode, and manipulate multimedia files. 
    - Importing Modules
      - train_test_split (from sklearn.model_selection)
To split datasets into training and testing subsets.
      - AudioSegment (from pydub)
The module enables loading audio files in multiple formats like WAV, MP4, W4A, etc.
  - Data Preprocess
    - MFCC
The audio files mentioned in the first bullet point were put into five categories; namely, 'up,' 'down,' ‘left,’ ‘right’ and ‘click.’ MFCC is implemented to extract data features in different classes and prepare the data for the machine learning model.
  - Model Training
    - Building CNN Model
      - Data Splitting
        The data was split into 20% training data and 80% testing data.
      - Data Scaling
        Features of audio clips that fall in the same category were first standardized by removing the mean and scaling to unit variance. “x_train” was scaled after being reshaped into a 2D array. After the procedure, they were reshaped back into 4D.
    - Defining CNN Model 
Below are the chronological layers we insert into the model.
      - First Conv2D Layer with relu activator
      - First MaxPooling2D Layer with relu activator
      - Second Conv2D Layer with relu activator
      - Second MaxPooling2D Layer with relu activator
      - Flatten Layer
To flatten the 3D output to 1D.
Dense Layers with relu activator
      - Dropout Layer (set at 0.5)
To prevent overfitting. (More the solution to overfitting will be discussed in the “Training Process” section below.)
      - Dense Layers with softmax activator
Softmax activation in the final layer produces output probabilities for each class.
    - Compiling the Model
The elements we utilize to compile the model and some brief explanations or introductions are as below.
      - Optimizer: adam. An adaptive learning rate optimization algorithm.
      - Loss: sparse_categorical_crossentropy. To Classify tasks with integer labels.
      - Metrics: accuracy. To evaluate the model's performance.
    - Training Process
      - Early Stopping
An early stop mechanism with the monitor of value loss was added to the training process after encountering overfitting. Overfitting refer to the situation when the model gave too accurate predictions for training data, but not for new data. To ensure the desirable performance of the model for actual application, a  dropout layer is added (more data were also collected to alleviate the condition).
      - Model Fit
After numerous attempts, a more preferable setting came out to be epochs=20; batch size=64; validation split=0.3 (30% of the training data will be set aside as the validation set, and the model will train on the remaining 70% of the data.)
  - Cursor Control
After training the model to distinguish between the five classes, we combine it with the cursor positioning Python module, PyAutoGui to use voice commands to control the mouse. 
Locations on the screen are referred to by X and Y Cartesian coordinates. The X coordinate starts at 0 on the left side and increases going right, while the Y coordinate starts at 0 on the top and increases going down (a reverse way as mathematic). 
    - Cursor Movement
After setting the screen resolution size, (2560, 1440) the moving distance each time is 400 pixels. 
    - Voice Command Reception
The period between each voice reception is 1 second. If nothing is spoken during the duration of the model recognizes the sound recorded as words other than the five, the model will return an error and no action will be taken. 

## Code


## Results
### Confusion Matrix
The following confusion matrix displays the results of our model's predictions compared to the actual classes. It indicates that our model is better at distinguishing "up," "down," and "click" commands, but has more difficulty distinguishing between "left" and "right."  
<img width="358" alt="image" src="https://github.com/JoeChuang02/Cursor-Mover/assets/174952737/8c167a0d-699f-4b39-a708-d559fdb8d60c">


## Future Prospect
### Future Research and Development
The future of voice command recognition is incredibly promising, with continuous improvements in accuracy and responsiveness expected as technology advances. Our project lays the groundwork for further exploration and enhancement in this field. Future research could focus on:
1. Improving Recognition Accuracy: By incorporating more diverse and extensive datasets, we can improve the system's ability to recognize various accents and tones, making it more robust and reliable.
2. Expanding Command Vocabulary: Adding more commands and improving the system's ability to understand complex instructions can broaden its application scope.
3. Integration with AI and IoT: Combining voice recognition with artificial intelligence and Internet of Things (IoT) technologies can create smarter, more responsive environments.

### Potential Applications and Innovations
The implications of our research extend far beyond the immediate application. Here are several promising areas where this technology can make a significant impact:
1. Enhanced Accessibility:
Voice command systems can revolutionize how people with disabilities interact with technology. This can lead to more inclusive designs in consumer electronics, home automation, and public services.
2. Smart Home Integration:
In the realm of smart homes, voice recognition can provide seamless control over various devices, from lights and thermostats to security systems and appliances, enhancing convenience and energy efficiency.
3. Autonomous Vehicles:
Voice commands can enhance safety and usability in autonomous vehicles by allowing hands-free control of navigation, entertainment, and climate settings, and by providing critical command inputs in emergencies.
4. Human-Machine Interaction:
As voice command technology becomes more intuitive, it can reduce the reliance on traditional input devices like keyboards and mice, leading to more natural and efficient human-machine interactions in various settings, from personal computing to industrial operations.
5. Personalized User Experiences:
Advanced voice recognition systems can be tailored to recognize individual voices and preferences, creating highly personalized experiences in customer service, entertainment, and daily activities.




## Contributors
112ZU1002 莊鎧肇 Joe (Project Manager, Programmer, Final Paper Research Methods and Result Compose)  
112ZU1004 方喻麟 Teresa (, Final Paper Research Methods and Result Compose)  
112ZU1009 何靜書 Cecilia (Slides Designer, Data Collector, Final Paper Motivation and Future Prospect Compose)  
112Zu1010 曹紫茵 Ariel (Slides Content Organizer, Data Collector, Final Paper Concept Compose)  
112ZU1037 羅怡蕊 Monse (Poster Editor, Data Collector)  
112ZU1052 賴珮妤 Ashley (Poster Copyeditor, Data Collector, Final Paper Concept Compose)

## Acknowledgments

[Thank any individuals or organizations who provided support or assistance during your project, including funding sources or data providers.]

## References

[List any references or resources that you used during your project, including data sources, analytical methods, and tools.]
