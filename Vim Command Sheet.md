|COMMAND|DESCRIPTION|
|-----------|-----------|
|**vim filename**|To start Vim from the shell prompt|
|**:q**|To close the help window.|
|**:q!**|Exit vim Without Saving|
|**:wq**|Exit vim by save the changes.|
|**x**|To delete the character at the cursor|
|**i**|Insert before the cursor|
|**a**|To append text after the line|
|**dw**|To delete from the cursor up to the next word.|
|**de**|To delete from the cursor up to the end of the word.|
|**d$**|To delete from the cursor to the end of a line.|
|**dd**|To delete a whole line.|
|**2w**|To repeat a motion prepend it with a number|
|**0**|To move to the start of the line use a zero|
|**u**|To undo previous actions.|
|**U**|To undo all the changes on a line.|
|**CTRL-R**|Redo|
|**p** |To put back text that has just been deleted.|
|**r**|To replace the character under the cursor.|
|**ce**|To change from the cursor to the end of the word.|
|**c$**|To change to the end of a line.|
|**CTRL-G**|Displays your location in the file and the file status.|
|**gg** |Moves to the first line.|
|**G** |Moves to the end of the file.|
|**number G**|To Moves to that line number.|
|**/phrase**|To searches FORWARD for the phrase.|
|**?phrase**|To searches BACKWARD for the phrase.|
|**n**|To find the next occurrence in the same direction(After a search type).
|**N**|To search in the opposite direction.|
|**CTRL-O**|Takes you back to older positions.|
|**CTRL-I**|To newer positions.|
|**%**|the cursor is on a (,),[,],{, or } goes to its match.|
|**:s/old/new**|To substitute new for the first old in a line.|
|**:s/old/new/g**|To substitute new for all 'old's on a line.|
|**:#,#s/old/new/g** |To substitute phrases between two line #'s|
|**:%s/old/new/g**|To substitute all occurrences in the file.|
|**:%s/old/new/gc** |To ask for confirmation each time add 'c'.|
|**:!command**|To executes an external command.|
|**:w FILENAME**|writes the current Vim file to disk with name FILENAME.|
|**v  motion :w FILENAME**|saves the Visually selected lines in file|
|**:r FILENAME**|retrieves disk file FILENAME and puts it below the cursor position.|
|**:r !dir**|reads the output of the dir command and puts it below the cursor position.|
|**o**|To open a line BELOW the cursor and start Insert mode.|
|**O**|To open a line ABOVE the cursor.|
|**a**|To insert text AFTER the cursor.|
|**A**|To insert text after the end of the line.|
|**e**|command moves to the end of a word.|
|**y**|Operator yanks (copies) text,  
|**p** |puts (pastes) it.|
|**r**|To replace the character under the cursor.|
|**R**| Enters Replace mode until  ESC  is pressed.|
|**:set xxx**|Sets the option "xxx".Some options are: **ic** 'ignorecase' ignore upper/lower case when searching **is** 'incsearch'	show partial matches for a search phrase **hls** 'hlsearch'	highlight all matching phrases You can either use the long or the short option name.|
|**:set noic**|Prepend "no" to switch an option off|
|**:help**| To open a help window.|
|**:help cmd**|To find help on cmd .|
|**CTRL-W**|To jump to another window.|
|**CTRL-D**|To see possible completions.|
|**TAB**|To use one completion.|
