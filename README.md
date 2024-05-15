NOTE: This was my senior design project in December 2022 prior to any work experience. As of May 15th, 2024, the code and project has not been updated nor has it been improved. This is the raw submitted senior design project that won an award for the class of 2022 Computer Engineerings at FIU College of Engineering and Computing.

# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Installing Dependencies ](#installing-dependencies-)
  - [Running the Project](#running-the-project)
  - [Other Functionalities](#other-functionalities)
    - [More Predictions Button](#more-predictions-button)
    - [Implementing BUY/SELL Buttons](#implementing-buysell-buttons)
  - [Repository File Explanations](#repository-file-explanations)

<br>

## Installing Dependencies <a name="Introduction"></a>
1. In order to run this program without errors, all the necessary libraries must be installed as outlined per [Requirements.txt](Requirements.txt) file. For the easiest installation process: 
   - Open this project in [VS Code](https://code.visualstudio.com/download) and open the terminal by pressing 
``
Ctrl+Shift+`
``.
   - Then, install the dependecies with pip and the terminal using:
  
     ```
     pip3 install -r Requirements.txt
     ```
<br>

## Running the Project
1. Once all dependencies have been installed, you are ready to run the project.
    - **Using VS Code:** Navigate to the project's [RunMe.py]([##RunMe.py](RunMe.py)) file, and run the file. 
        
        **NOTE:** There will be a 5 to 10 minute delay after you type your stock in the first window in order to train and test the algorithms.
        <br>

        ### First Pop-Up Window
        - Contains entry box for user to input stock. Once typed, and enter is selected, algorithms will begin computing.
  
          ![](Images/FirstWindow.png)
          
          <br>

        ### Second Pop-Up Window
        - Here, you can select which algorithm you would like to see the prediction for.
        
            **Dark Mode**
          ![](Images/DarkSecondWindow.png)

          <br>

        ### Algorithmic Predictions Windows
        - Lastly, you can now access all the information relevant to the learner you selected.
 
            **Deep-Q Algorithm (Light Mode)**
          ![](Images/LightLastWindow.png)  
          <br>

## Other Functionalities

### More Predictions Button
1. Once the second window pops up and you select a learner, there is a `More Predictions` button on the bottom right.
    - The `More Predictions` button brings another window to pop up with a plot of approximately the last 30 days of predictions. 
      - The green bar represents an `ACCURATE` prediction, meaning the following day there was a positive portfolio return based on the prediciton. 
      - A red bar represents the opposite, `INACCURATE`. 
    - The x-axis shows integers with a negative sign preceding it. `-1` means yesterday, `-4` means 4 days ago, `-20` means 20 days ago, etc.
  
        ![](Images/TslaStlPredictionsPlot.png)

<br>

### Implementing BUY/SELL Buttons
1. The user interface has two buttons after selecting the learner: BUY and SELL. Currently, this project as is does not incorporate BUY and SELL of the stock you inputted in the fist pop-up window.

    - If you **do not** wish to use the BUY and SELL buttons functionality, ignore this section and [run the project](#running-the-project)! Otherwise, follow these instructions:
    
        **Step One:** Install [Trader WorkStation (TWS) API](https://www.interactivebrokers.com/en/trading/tws.php#tws-software).
        
        **Step Two:** Create an [InteractiveBrokers account](https://gdcdyn.interactivebrokers.com/Universal/Application) and ensure a funded account.

        **Step Three:** Un-comment out lines 562-564, 568-570, 951-953, and 957-959 of [UserInterface.py](Interfaces/UserInterface.py).

        **Step Four:** Follow short instructions on `Connecting Code to TWS` section of the [Software Documentation.docx](University/Software%20Documentation.docx) to ensure an established Interactive Brokers connection.

        **Step Five:** [Run the project](#running-the-project)!

## Repository File Explanations
Refer to the [Software Documentation.docx](University/Software%20Documentation.docx) that provides more in-depth descriptions of all code in this project. This was a senior design project in which all programming and documentation of code was done by Daniel Leal. 

However, the documentation was done prior to any work experience so it is not at the same level of documentation that I can provide now. Nonetheless, I left this repository here to show growth as this was my first time tying AI to a project.
