import re
import csv
import os

field1='title'
field2='tableofcontents'                     
field3='abstract'
path='sorted_title'
count=0

def extends_classification(prev_ddc, new_ddc):
    prev_ddc=int(prev_ddc)
    new_ddc=int(new_ddc)
    if prev_ddc/100!=new_ddc/100:
	return -1
    else:
	if prev_ddc/10!=new_ddc/10 and prev_ddc/10==0:
	    return 2                                              #2nd level of classification added
	elif prev_ddc/10!=new_ddc/10 and prev_ddc/10!=0:
	    return -1 	 	 
	else:
	    if prev_ddc!=new_ddc and prev_ddc%10==0:
	 	return 3                                          #3rd level of classification added
	    elif prev_ddc!=new_ddc and prev_ddc%10!=0:
	 	return -1 	 	 

         	 	 
def main():
    csv.field_size_limit(1024*1024*512)
    '''
    isolator("IEEE_title1.csv")
    '''
    for files in os.listdir(path):
        if files.endswith(".csv"):
	    isolator(files)
    print count
            
def isolator(filename):
    print filename
    inputF=open(path+"/"+filename,"rb")
    filename=filename.strip('.csv')
    reader=csv.reader(inputF)
    outputF=open("abs_contents_title_primary_1/"+filename+".csv","wb")
    writer=csv.writer(outputF)
    field2Idx=-1
    field3Idx=-1
    writer.writerow(['ddc_level3','classification_level',field1,field2,field3])
    
    newRow=[]
    header=next(reader)
    for item in range(0,len(header)):
        if 'id'== header[item]:
            idIdx=item
        if 'ddc' in header[item]:
            ddcIdx=item
        if field1 in header[item]:
            field1Idx=item
        if field2 in header[item]:
            field2Idx=item
	if field3 in header[item]:
            field3Idx=item

     #print field2Idx
    for row in reader:
        flag=0
        #idData=row[idIdx]
        ddcData=row[ddcIdx]
        field1Data=row[field1Idx]
        if(field2Idx>=0):
            field2Data=row[field2Idx]
        else:
            field2Data=''
        if(field3Idx>=0):
            field3Data=row[field3Idx]
        else:
            field3Data=''
        global count
        ddc_3=[]
        classification_level=[]
        ddcList=re.findall("\d\d\d::",ddcData) #\d*[\.]?\d*:: to check for decimal-containing classifications
        print ddcList
        pre=None
        pre_level=-1
        level=-1
        for item in ddcList:
            item=int(item.strip(':'))
            print "item",item
            if pre is None:
                pre=int(item)
            pre_level=level
 	    level=extends_classification(pre,item)
            print "pre_level",pre_level
            print "level",level
 	    if level<0:
 		ddc_3.append(pre)
 		classification_level.append(pre_level)
            print "ddc_3",ddc_3
            pre=item
            exit()

        ddc_3.append(pre)
        classification_level.append(level)
        ddc_3=' '.join(str(format(x,'03d')) for x in ddc_3)
        classification_level=' '.join(str(x) for x in classificatio_level)
        writer.writerow([ddc_3,classification_level,field1Data,field2Data,field3Data]) 
        
        count+=1
    outputF.close()
    
        
main()    
    
        
        
    

