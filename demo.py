import tkinter as tk

class DemoWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize the demo window
        self.title('AI Detector Demo')
        self.geometry('400x250')

        # Create the grid layout
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        # Initialize widgets
        self.inputBox = tk.Text()
        self.inputBox = tk.Text(self, height=8, width=40)
        self.inputBox.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew")

        self.clearButton = tk.Button(self, text="Clear", command=self.clearInput)
        self.clearButton.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")

        # Add listeners
        self.inputBox.bind('<KeyRelease>', self.onInputRecieved)

        # Initialize text quality fields
        self.inputText = ''

        # Initialize timers
        self.refreshAnalysisTimer = None

    def clearInput(self):
        # Update widgets
        self.inputBox.delete("1.0", "end")

        # Update analysis variables
        self.inputText = ''

    def onInputRecieved(self, _):
        # Return if the content of the input box has not changed
        inputBoxContent = self.inputBox.get('1.0', 'end-1c')
        if (inputBoxContent == self.inputText):
            return

        # Update the input text
        self.inputText = inputBoxContent

        # Queue refreshing the analysis
        if self.refreshAnalysisTimer is not None:
            self.after_cancel(self.refreshAnalysisTimer)
        self.refreshAnalysisTimer = self.after(500, self.refreshAnalysis)

    def refreshAnalysis(self):
        # Reset analysis timer
        self.refreshAnalysisTimer = None

        # Evaluate text qualities

        # Feed to model

        print('Analysis refreshed')
        


def runDemo():
    demoWindow = DemoWindow()
    demoWindow.mainloop()

if __name__ == '__main__':
    runDemo()