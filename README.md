<h1>Drowsy Driver Detection System</h1>
<p>A system to detect drowsiness using extracted HOG facial landmarks. Developed as a part of CSE 4510: Software Development.</p>

<h2>Team Members</h2>
<ul>
<li>
Mashrur Mahmud Morshed<b><a href="https://github.com/ID56">(@ID56)</a></b>
</li>
<li>
Hasan Tanvir Iqbal<b><a href="https://github.com/TanvirHundredOne">(@TanvirHundredOne)</a></b>
</li>
<li>
Mazharul Islam Rishad<b><a href="https://github.com/Aporamithos">(@Aporamithos)</a></b>
</li>
</ul>

<h2>Features</h2>
<ul>
<li>Detects drowsiness from a real-time video stream</li>
<li>Calculates number of blinks of driver and shows on output display</li>
<li>Shows runtime information on display</li>
<li>Generates an excel file(.csv format) containing metadata:
  <ul>
		<li>Eye position per three seconds</li>
		<li>Blinks per ten seconds</li>
		<li>Drowsiness event timestamp</li>
		<li>System start and end timestamp</li>
		<li>Effective frame rate</li>
	 </ul>
  </li>
	</ul>

<h2>Current Status</h2>
	Implementation done on a desktop scale. Will implement in raspberry pi for portability.

<h2>Frameworks/Libraries Used</h2>
	<ul>
		<li>Python</li>
		<li>OpenCV</li>
		<li>DLIB</li>
		<li>Anaconda distribution</li>
	</ul>

<h2>Notes</h2>
	After installing dependencies, download the pre-trained landmark predictor from the <a href="https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2">dlib github repository</a>, and keep in the same directory.

<h2>Credits</h2>
	-<a href="https://github.com/davisking">Davis King</a>, creator of the dlib library.<br>
	-<a href="https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/">The Pyimagesearch BLog</a>, for it's neat drowsiness tutorials.
