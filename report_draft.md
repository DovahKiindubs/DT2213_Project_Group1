# Polyphonic Finger-Keyboard FM Formant Synth

## Introduction / Idea / Problem

## Background / Theory

The instrument is based on gesture-controlled musical interaction. Instead of using a normal keyboard, mouse, or MIDI controller, the performer uses body movement as part of the performance. This makes the instrument more visual and physical, and allows musical parameters to be controlled in a continuous way.

The technical background combines computer vision and sound synthesis. Computer vision is used to estimate hand and mouth movement from a webcam image. These movements are then mapped to musical controls such as pitch, timbre, brightness, and vowel shape.

For sound generation, the project uses FM synthesis and formant filtering. FM synthesis can create many different electronic timbres by changing the relationship between a carrier oscillator and a modulator oscillator. Formant filtering is inspired by the human voice, where different vowels are shaped by resonant frequency areas.

The pitch system is based on a major pentatonic scale. This scale was chosen because it is simple and stable: many note combinations sound consonant, which is useful for an experimental gesture interface.

## Method

The system was implemented in Python. The webcam image is processed frame by frame using MediaPipe. For each frame, the program detects both hands and the face. One hand is used as the note hand and the other hand is used as the timbre and expression hand.

Finger bending is detected from hand landmark distances. For each finger, the program compares the distance from the wrist or palm to the fingertip with the distance to an inner finger joint. A bent finger gives a smaller ratio than a straight finger. Hysteresis thresholds are used so that notes do not rapidly turn on and off because of small tracking noise. When a finger bends, the corresponding voice is triggered. When it straightens, the voice is released.

The note frequencies are generated from a pentatonic scale. The note hand has five possible notes, one for each finger. The vertical distance between the note hand and the timbre hand selects the register. If the note hand is higher than the timbre hand, the instrument plays in a higher register. If it is lower, it plays in a lower register. The register is read when a note is triggered, so held notes remain stable even if the hands move afterward.

The left-hand gesture selects one of five timbre presets. These presets are pad, key, glass, drone, and rift. Each preset changes the amplitude envelope, FM modulation envelope, oscillator ratios, harmonic layers, filter mix, gain, and other sound parameters. The rift preset also uses a saw wave layer and a low-pass filter for a retro synth bass character.

The horizontal distance between the two hands controls brightness. When the hands are close together, the modulation depth is low and the sound is darker. When the hands move apart, the modulation depth increases and the sound becomes brighter.

Mouth movement controls the formant filters. The face model estimates jaw opening, smile, and pucker values. Jaw opening is mapped to the first formant frequency, while smile and pucker are mapped to the second formant frequency. The output of all voices is summed, passed through the formant filters, and sent to the audio output in real time using the `sounddevice` library.
