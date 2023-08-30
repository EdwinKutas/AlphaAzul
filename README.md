# AlphaAzul

This is an implementation of the AlphaGo algorithm for the board game Azul. This is a non detereministic board game, so the original Montecarlo tree search method has been modified. Data is artificially generated to resemble the results from the end of a round and then trained from there for one round. The current status of the code is that it works, howerver due to speed issues it takes a very long time to run.
The current list of things to implement:
  - Change the way the game outputs its states, at the moment it does this as tensors which slows things down.
  - Reimpliment the way the game models taking pieces, at the moment this is a big slow down.
  - Implement a cleaner API for play and graphics.
