#dimensionamento simplificado 


#Terminal de passageiros
#Dados necessários
Populacao = input("Quantas pessoas habitam na cidade:")
Anos = input( "Perpectiva de validade do projeto:")
Porcentagem_horario_Pico= 0.0005
Fator_de_segurança= 1.5
Area_Confortavel_Pessoa=2
Crescimento_populacional_anual=0.0052

#Calculo do tamanho do terminal
Tamanho_do_terminal= float(Porcentagem_horario_Pico)*int(Area_Confortavel_Pessoa)*float(Fator_de_segurança)*(int(Populacao)+int(Populacao)*float(Crescimento_populacional_anual)*int(Anos))
print(Tamanho_do_terminal)


#Tamanho da pista de pouso
Altitude = input("Altitude da cidade:")
Temperatura= input("Temperatura media:")
Declividade= input("Declividade media:")
L0= 1200
print(Tamanho_Final_Pista)
