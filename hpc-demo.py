# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'first-draft.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import sys
from sklearn.preprocessing import LabelEncoder
import os


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 644)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("QWidget#centralwidget{background-image: url(images/5_cyber_trends_landing_page.jpg);}\n"
"\n"
"\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        self.button1 = QtWidgets.QPushButton(self.centralwidget)
        self.button1.setGeometry(QtCore.QRect(300, 200, 141, 31))
        self.button1.setStyleSheet("border-radius:20px;\n"
"font: 12pt \"MS Shell Dlg 2\";\n"
"color:white;\n"
"background-color: rgb(38, 44, 54);")
        self.button1.setObjectName("button1")
        self.checkBoxBackdoor = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxBackdoor.setGeometry(QtCore.QRect(440, 130, 91, 41))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.checkBoxBackdoor.setFont(font)
        self.checkBoxBackdoor.setAutoFillBackground(False)
        self.checkBoxBackdoor.setStyleSheet("color:white;\n"
"border-radius:20px;\n"
"background-color: rgb(38, 44, 54);")
        self.checkBoxBackdoor.setObjectName("checkBoxBackdoor")
        self.checkBoxTrojan = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxTrojan.setGeometry(QtCore.QRect(220, 130, 91, 41))
        self.checkBoxTrojan.setStyleSheet("color:white;\n"
"border-radius:20px;\n"
"background-color: rgb(38, 44, 54);")
        self.checkBoxTrojan.setObjectName("checkBoxTrojan")
        self.checkBoxVirus = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxVirus.setGeometry(QtCore.QRect(330, 130, 91, 41))
        self.checkBoxVirus.setStyleSheet("color:white;\n"
"border-radius:20px;\n"
"background-color: rgb(38, 44, 54);")
        self.checkBoxVirus.setObjectName("checkBoxVirus")
        self.checkBoxWorm = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxWorm.setGeometry(QtCore.QRect(550, 130, 91, 41))
        self.checkBoxWorm.setStyleSheet("color:white;\n"
"border-radius:20px;\n"
"background-color: rgb(38, 44, 54);\n"
"text-align:right;\n"
" ")
        self.checkBoxWorm.setObjectName("checkBoxWorm")
        self.checkBoxRootkit = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxRootkit.setGeometry(QtCore.QRect(110, 130, 91, 41))
        self.checkBoxRootkit.setStyleSheet("border-radius:20px;\n"
"color:white;\n"
"background-color: rgb(38, 44, 54);")
        self.checkBoxRootkit.setObjectName("checkBoxRootkit")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(100, 250, 531, 261))
        self.textEdit.setStyleSheet("background-color: rgb(38, 44, 54);\n"
"background-color: rgb(33, 38, 47);\n"
"color:rgb(255, 255, 255);\n"
"border-radius:20px;"
"text-align:center;\n"
"padding-left:90%;")
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 20, 411, 41))
        self.label.setStyleSheet("color:white;\n"
"font: 18pt \"MS Shell Dlg 2\";\n"
"")
        self.label.setObjectName("label")
        self.button2 = QtWidgets.QToolButton(self.centralwidget)
        self.button2.setGeometry(QtCore.QRect(480, 200, 51, 31))
        self.button2.setStyleSheet("color:white;\n"
"background-color: rgb(38, 44, 54);\n"
"border-radius:20px;")
        self.button2.setObjectName("button2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 740, 27))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.datasetMalware2 = None
        self.button2.clicked.connect(self.chooseFile)
        self.button1.clicked.connect(lambda: self.checkboxes(self.datasetMalware2))
        
    
    def chooseFile(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, "Open The File", os.getenv(
                "HOME"))  # This will return tuple first element contains the file
        print(file_name[0])
        self.datasetMalware2 = pd.read_csv(file_name[0])
    
    def checkboxes(self, datasetMalware):
        malwareColumns = list(datasetMalware.columns)
        newDf = pd.DataFrame(columns = malwareColumns)
        if self.checkBoxBackdoor.isChecked(): #0
            #print(datasetMalware.loc[datasetMalware['class']=='worm'][1:8])
            newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='backdoor'])
        if self.checkBoxRootkit.isChecked(): 
            newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='rootkit'])
        if self.checkBoxTrojan.isChecked(): 
            newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='trojan'])
        if self.checkBoxVirus.isChecked():
            newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='virus'])
        if self.checkBoxWorm.isChecked():
            newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='worm'])
            
        #newDf.to_csv('D:\\PYQTCYBER\\shotgun.csv',index=False) 
        newDf = newDf.append(datasetMalware.loc[datasetMalware['class']=='benign'])
        self.testMalware(newDf)
        
        
    def testMalware(self, Malware):
        Malware.head()
        
        Malware.isnull().any()
        Malware['instructions']=Malware['instructions'].fillna(Malware['instructions'].mean())
        Malware.describe()
        
        
        le_color = LabelEncoder()
        Malware['class'] = le_color.fit_transform(Malware['class'])
        Malware['class'].unique()
        
        #correlation plot
        numeric_data = Malware.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        sns.heatmap(corr)
        
        plt.figure(figsize=(12,6))
        sns.heatmap(Malware.corr(),annot=True)
        
        #Removal of outliers by interquartile range method
        Q1 = Malware.quantile(0.25)
        Q3 = Malware.quantile(0.75)
        IQR = Q3 - Q1
        
        Malware=Malware[~((Malware < (Q1 - 1.5 * IQR)) |(Malware > (Q3 + 1.5 * IQR))).any(axis=1)]
        Malware.shape
        
        
        #removing multicollinearity by variance inflation factor method
        y = Malware['class']
        Malware = Malware.drop(columns = ['class'])
        Malware.head()
        
        Malware = Malware.astype(np.int64)
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        [variance_inflation_factor(Malware.values, j) for j in range(Malware.shape[1])]
        thresh = 5.0
        output = pd.DataFrame()
        k = Malware.shape[1]
        vif = [variance_inflation_factor(Malware.values, j) for j in range(Malware.shape[1])]
        for i in range(1,k):        
            a = np.argmax(vif)        
            if vif[a] <= thresh :
                break
            if i == 1 :          
                output = Malware.drop(Malware.columns[a], axis = 1)
                #print(output)
                vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
            elif i > 1 :
                output = output.drop(output.columns[a],axis = 1)
                #print(output)
                vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        train_out = output
        Malware = Malware.drop(columns=train_out.columns, axis=1)
        Malware.info()
        #train_out.info()  
        Malware['class'] = y
        
        print(Malware.info())

        print("***********")
        print(Malware['class'])
        print("***********")

        # Separating out the features
        X = Malware.iloc[:, :-1].values  
        
        # Separating out the target
        y = Malware.iloc[:,-1].values
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        # Standardizing the features
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
        #fitting random forest classifier to the training dataset
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(max_depth=10, random_state=0)
        classifier.fit(X_train_res, y_train_res.ravel())
        y_pred = classifier.predict(X_test)
        
        from sklearn.metrics import confusion_matrix  
        from sklearn.metrics import accuracy_score
        # from sklearn.metrics import recall 
        
        from sklearn.metrics import classification_report
        actualReport = ""
        report = classification_report(y_test, y_pred)
        actualReport += report
        print(report)
        
        cm = confusion_matrix(y_test, y_pred)
        #actualReport += report
        print(cm)
        actualReport += "Accuracy " + str(accuracy_score(y_test, y_pred))
        print('Accuracy' + str(accuracy_score(y_test, y_pred))) 
        # Visualising the Test set results
        
        actualReport += "\n"
        
        classifier.fit(X_train_res, y_train_res.ravel())
        
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        treeclassifier = DecisionTreeClassifier(criterion = 'gini', random_state = 8, splitter = 'random')
        treeclassifier.fit(X_train_res, y_train_res.ravel())
        
        Y_pred = treeclassifier.predict(X_test)
        from sklearn.metrics import confusion_matrix  
        from sklearn.metrics import accuracy_score
        # from sklearn.metrics import recall 
        
        cm = confusion_matrix(y_test, Y_pred)  
        print(cm)  
        actualReport += "Accuracy " + str(accuracy_score(y_test, Y_pred))
        print('Accuracy' + str(accuracy_score(y_test, Y_pred))) 
        
        actualReport += "\n"
        
        from sklearn.metrics import classification_report
        
        report = classification_report(y_test, Y_pred)
        actualReport += report
        print(report)
        self.textEdit.setText(actualReport)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button1.setText(_translate("MainWindow", "Run"))
        self.checkBoxBackdoor.setText(_translate("MainWindow", "  Backdoor"))
        self.checkBoxTrojan.setText(_translate("MainWindow", "  Trojan"))
        self.checkBoxVirus.setText(_translate("MainWindow", "  Virus"))
        self.checkBoxWorm.setText(_translate("MainWindow", "  Worm"))
        self.checkBoxRootkit.setText(_translate("MainWindow", "  Rootkit"))
        self.label.setText(_translate("MainWindow", "HPC Malware Detection Demo"))
        self.button2.setText(_translate("MainWindow", "..."))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


