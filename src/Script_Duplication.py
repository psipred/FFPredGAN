__author__ = 'cenwan'

#-----Please download the 'GOTerm_List.txt' file via http://bioinfadmin.cs.ucl.ac.uk/downloads/FFPredGAN/

with open(".../GOTerm_List.txt") as ScriptFile:
    GOTermsRaw=ScriptFile.readlines()

GOTerms=[]
for index in range(0,len(GOTermsRaw)):
    GOTerms.append(GOTermsRaw[index].strip())

#-----Please download the 'Generating_Synthetic_Positive_Samples_FFPred-GAN.py' file via https://github.com/psipred/FFPredGAN/blob/master/src/
with open(".../Generating_Synthetic_Positive_Samples_FFPred-GAN.py") as ScriptFile:
    Script=ScriptFile.readlines()

for number in range(1,302):
    GOTermLabel=GOTerms[number-1]
    jobScriptWritter = open('.../'+str(number)+".py",'w')

    for rowIndex in range(len(Script)):
        if "GO0034613" in Script[rowIndex]:
               jobScriptWritter.write("GOTerm='"+GOTermLabel+"'")
               jobScriptWritter.write("\n")
        else:
            jobScriptWritter.write(Script[rowIndex])
    jobScriptWritter.flush()
    jobScriptWritter.close()
