import csv
import os

def main():
    csv.field_size_limit(1024*1024*512)
    statsF=open("sorted_abs/stats.csv","wb")
    writer1=csv.writer(statsF)
    header=['file name','# with ddc & title', '# with ddc & abstract' ,'# with ddc & tableofcontents']
    writer1.writerow(header)
    '''row=sorter("Annotated.csv")#AllCSV/
    writer1.writerow(row)
    '''
    for file in os.listdir("AllCSV"):
        row=sorter(file)
        print row
        writer1.writerow(row)
           

    
def sorter(filename):  #MAIN FUNCTION FOR THE WORK

    ddcList=[]
    ddcFlag=0
    titleList=[]
    titleFlag=0
    abstractList=[]
    abstractFlag=0
    tableofcontentsList=[]
    tableofcontentsFlag=0
    fileCountTitle=1         #no. of files created for this file having entries with non-blank title fields
    fileCountAbs=1
    fileCountTable=1
    noWithddc=0               #no. of files with ddc/title present
    noWithabs=0
    noWithcontents=0

    inputF=open("AllCSV/"+filename,"rb") 
    filename=filename.strip(".csv")                #the filename given by the listdir function does NOT consist of the complete address

    reader=csv.reader(inputF)
  
    header=next(reader)
    idx=-1
    for item in header:
        idx+=1
        if ".ddc" in item:
            ddcList.append(idx)
        elif "title" in item:
            titleList.append(idx)
        elif ".abstract" in item:
            abstractList.append(idx)
        elif "tableofcontents" in item:
            tableofcontentsList.append(idx)

    if len(ddcList)==0 :
        return[filename,0,0,0]
    if len(ddcList)>1 :
        ddcFlag=1
    if len(titleList)>1 :
        titleFlag=1
    if len(abstractList)>1 :
        abstractFlag=1
    if len(tableofcontentsList)>1 :
        tableofcontentsFlag=1

    if(ddcFlag==1):
        #Checking whether the found cols are subsequet. Seek user's command, if not
        if( not areSubsequent(ddcList)):
            print("ATTENTION REQUIRED:\nfile %s: ddc found at"%filename),
            print ddcList,
            print ": ",
            for obj in ddcList:
                print header[obj],
           
            print "\nMerge?[Y/N]"
            ans=input()
            if not(ans=='y' or ans=='Y'):
                return [filename]
    

    if(titleFlag==1):
        #Checking whether the found cols are subsequet. Seek user's command, if not
        if( not areSubsequent(titleList)):
            print("ATTENTION REQUIRED:\nfile %s:title found at"%filename),
            print titleList,
            print ": ",
            for obj in titleList:
                print header[obj],
            print "\nMerge?[Y/N]"
            ans=input()
            if not(ans=='y' or ans=='Y'):
                return [filename]
    
        
    if(tableofcontentsFlag==1):
        #Checking whether the found cols are subsequet. Seek user's command, if not
        if( not areSubsequent(tableofcontentsList)):
            print("ATTENTION REQUIRED:\nfile %s:tableofcontents found at"%filename),
            print tableofcontentsList,
            print ": ",
            for obj in tableofcontentsList:
                print header[obj],
            print "\nMerge?[Y/N]"
            ans=input()
            if not(ans=='y' or ans=='Y'):
                return [filename]
    
        
    if(abstractFlag==1):
        #Checking whether the found cols are subsequet. Seek user's command, if not
        if( not areSubsequent(abstractList)):
            print("ATTENTION REQUIRED:\nfile %s:abstract found at"%filename),
            print abstractList,
            print ": ",
            for obj in abstractList:
                print header[obj],
            print "\nMerge?[Y/N]"
            ans=input()
            if not(ans=='y' or ans=='Y'):
                return [filename]
    
        
    #Merging col headers if required    
    newheader=[]
    toAvoid=[]
    for idx in ddcList[1:]:toAvoid.append(idx)
    for idx in titleList[1:]:toAvoid.append(idx)
    for idx in abstractList[1:]:toAvoid.append(idx)
    for idx in tableofcontentsList[1:]:toAvoid.append(idx)

    for idx in range(0,len(header)):
        if not(idx in toAvoid):
            newheader.append(header[idx])
            if len(ddcList) and idx==ddcList[0]:
                ddcHeader=len(newheader)-1
                header[ddcHeader]='ddc'
            if len(titleList) and idx==titleList[0]:
                titleHeader=len(newheader)-1
                header[titleHeader]='title'
            if len(tableofcontentsList) and idx==tableofcontentsList[0]:
                tableofcontentsHeader=len(newheader)-1
                header[tableofcontentsHeader]='tableofcontents'
            if len(abstractList) and idx==abstractList[0]:
                abstractHeader=len(newheader)-1
                newheader[abstractHeader]='abstract'

    header=newheader

    
        
        

    outputF_title=open("sorted_title/"+filename+"_title"+str(fileCountTitle)+".csv","wb")
    outputF_abstract=open("sorted_abs/"+filename+"_abstract"+str(fileCountAbs)+".csv","wb")
    outputF_contents=open("sorted_contents/"+filename+"_contents"+str(fileCountTable)+".csv","wb")

    writer_title=csv.writer(outputF_title)
    writer_title.writerow(header)
    writer_abs=csv.writer(outputF_abstract)
    writer_abs.writerow(header)
    writer_contents=csv.writer(outputF_contents)
    writer_contents.writerow(header)

    print "Processing: "+filename    
    for row in reader:
        ddcIdx=None
        titleIdx=None
        absIdx=None
        contentsIdx=None
        
        #Merging cols if required
        if(ddcFlag==1):
            for idx in range(1,len(ddcList)):
                row[ddcList[0]]=row[ddcList[0]]+" "+row[ddcList[idx]]
            row[ddcList[0]]=row[ddcList[0]].strip()
        ddcIdx=ddcList[0] if len(ddcList)>0 else None

        #Merging cols if required
        if(titleFlag==1):
            for idx in range(1,len(titleList)) :
                row[titleList[0]]=row[titleList[0]]+" "+row[titleList[idx]]
            row[titleList[0]]=row[titleList[0]].strip()
        titleIdx=titleList[0] if len(titleList)>0 else None
        
        #Merging cols if required
        if(abstractFlag==1):
            for idx in range(1,len(abstractList)) :
                row[abstractList[0]]=row[abstractList[0]]+" "+row[abstractList[idx]]
            row[abstractList[0]]=row[abstractList[0]].strip()
        absIdx=abstractList[0] if len(abstractList)>0 else None
        
            
        #Merging cols if required
        if(tableofcontentsFlag==1):
            for idx in range(1,len(tableofcontentsList)) :
                row[tableofcontentsList[0]]=row[tableofcontentsList[0]]+" "+row[tableofcontentsList[idx]]
            row[tableofcontentsList[0]]=row[tableofcontentsList[0]].strip()
        contentsIdx=tableofcontentsList[0] if len(tableofcontentsList)>0 else None

        #Deleting the columns that have been merged to other
        newRow=[]
        toAvoid=[]
        for idx in ddcList[1:]:toAvoid.append(idx)
        for idx in titleList[1:]:toAvoid.append(idx)
        for idx in abstractList[1:]:toAvoid.append(idx)
        for idx in tableofcontentsList[1:]:toAvoid.append(idx)

        for idx in range(0,len(row)):
            if not(idx in toAvoid):
                newRow.append(row[idx])
                if idx==ddcIdx:
                    ddcIdx=len(newRow)-1
                if idx==titleIdx:
                    titleIdx=len(newRow)-1
                if idx==contentsIdx:
                    contentsIdx=len(newRow)-1
                if idx==absIdx:
                    absIdx=len(newRow)-1

        row=newRow
        
        if (ddcIdx and row[ddcIdx].strip()) :
            if  (titleIdx and row[titleIdx].strip()):
                writer_title.writerow(row)
                noWithddc+=1
            if noWithddc==(fileCountTitle*100000) :  #writing only a hundred thousand entries in one output file
                outputF_title.close()
                fileCountTitle+=1
                outputF_title=open("sorted_title/"+filename+"_title"+str(fileCountTitle)+".csv","wb")
                writer_title=csv.writer(outputF_title)
                writer_title.writerow(header)
            
            if  (absIdx and row[absIdx].strip()):
                writer_abs.writerow(row)
                noWithabs+=1
            if noWithabs==(fileCountAbs*100000) :  #writing only a hundred thousand entries in one output file
                outputF_abstract.close()
                fileCountAbs+=1
                outputF_abstract=open("sorted_abs/"+filename+"_abstract"+str(fileCountAbs)+".csv","wb")
                writer_abs=csv.writer(outputF_abstract)
                writer_abs.writerow(header)
 
            if  (contentsIdx and row[contentsIdx].strip()):
                # print row[contentsIdx]
                writer_contents.writerow(row)
                noWithcontents+=1
            if noWithcontents==(fileCountTable*100000) :  #writing only a hundred thousand entries in one output file
                outputF_contents.close()
                fileCountTable+=1
                outputF_contents=open("sorted_contents/"+filename+"_title"+str(fileCountTable)+".csv","wb")
                writer_contents=csv.writer(outputF_contents)
                writer_contents.writerow(header)
    outputF_contents.close()
    outputF_title.close()
    outputF_abstract.close()
    inputF.close()
    return[filename,noWithddc,noWithabs,noWithcontents]



def areSubsequent(l):
    for idx in range(0,len(l)-1):
        if not (l[idx] == l[idx+1]-1):
            return False
    return True
        
                
        
main()
        
