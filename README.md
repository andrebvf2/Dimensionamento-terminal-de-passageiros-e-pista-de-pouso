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


#Tamanho da pista de pouso. Considerando que o comprimento de uma pista de pouso de um aeroporto de pequeno porte, em condições extremamente ideais, seja de 1200m
Altitude = input("Altitude da cidade:")
Temperatura= input("Temperatura local:")
Declividade= input("Declividade media:")
Fator_declividade = 1 + float(Declividade)/100
Temperatura_altitude= 15 - 0.0065*float(Altitude)
Fator_temperatura = 1 + 0.01*(float(Temperatura)-float(Temperatura_altitude))
Fator_altitude= 1 + (0.07*float(Altitude)/300) 
L0= 1200

#Calculo do comprimento da pista de pouso

Tamanho_Final_Pista = int(L0) * float(Fator_altitude) * float(Fator_temperatura) * float(Fator_declividade)
print(Tamanho_Final_Pista)
