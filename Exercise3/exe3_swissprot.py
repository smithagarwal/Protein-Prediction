# -*- coding: utf-8 -*-
"""
IMPORTANT!:
Before writing an email asking questions such as
'What does this input has to be like?' or 
'What return value do you expect?' PLEASE read our
exercise sheet and the information in this template
carefully.
If something is still unclear, PLEASE talk to your
colleagues before writing an email!

If you experience technical issues or if you find a
bug we are happy to answer your questions. However,
in order to provide quick help in such cases we need 
to avoid unnecessary emails such as the examples
shown above.
"""
import re

from Bio import SeqIO # Tip: This module might be useful for parsing... 

############ Exercise 3: SwissProt ##########
class SwissProt_Parser:

    PARSER = SeqIO

    def __init__( self, path, frmt='uniprot-xml' ):
        '''
            Initialize every SwissProt_Parser with a path to a XML-formatted UniProt file.
            An example file is included in the repository (P09616.xml).
            Tip: Store the parsed XML entry in an object variable instead of parsing it
            again & again ...
        '''

        self.sp_anno = list(self.PARSER.parse(path, frmt))

        # Parse the XML file once and re-use it in the functions below

    # 3.2 SwissProt Identifiers
    def get_sp_identifier( self ):
        '''
            Input: 
                self: Use XML entry which has been parsed & saved during object initialization 
            Return:
                Unique SwissProt identifier for the given xml file
        '''
        identifier = ""
        for record in self.sp_anno:
            identifier = record.id
        return identifier

    # 3.3 SwissProt Sequence length
    def get_sp_sequence_length(self):
        '''
            Input: 
                self: Use XML entry which has been parsed & saved during object initialization 
            Return:
                Return sequence length of the UniProt entry as an integer.
        '''

        seq_len = ""
        for records in self.sp_anno:
            seq_len = len(records.seq)
        return seq_len

    # 3.4 Organism 
    def get_organism( self ):
        '''
            Input: 
                self: Use XML entry which has been parsed & saved during object initialization 
            Return:
                Return the name of the organsim as stated in the corresponding field
                of the XML data. Return value has to be a string.
        '''

        organism = ''
        for records in self.sp_anno:
            organism = records.annotations['organism']
        return organism

    # 3.5 Localizations
    def get_localization( self ):
        '''
            Input: 
                self: Use XML entry which has been parsed & saved during object initialization 
            Return:
                Return the name of the subcellular localization as stated in the 
                corresponding field.
                Return value has to be a list of strings.
        '''

        localization = list()
        for records in self.sp_anno:
            localization = records.annotations['comment_subcellularlocation_location']
        return localization

    # 3.6 Cross-references to PDB
    def get_pdb_support( self ):
        '''
            Input: 
                self: Use XML entry which has been parsed & saved during object initialization 
            Return:
                Returns a list of all PDB IDs which support the annotation of the
                given SwissProt XML file. Return the PDB IDs as list.
        '''

        pdb_ids = list()
        for records in self.sp_anno:
            dxrefs = records.dbxrefs
        print(dxrefs)
        for item in dxrefs:
            if re.search(r"PDB:", item):
                pdb_ids.append(item[4:])
        return pdb_ids

def main():
    print('SwissProt XML Parser class')
    swiss = SwissProt_Parser("P09616.xml")
    swiss.get_pdb_support()
    return None


if __name__ == '__main__':
    main()