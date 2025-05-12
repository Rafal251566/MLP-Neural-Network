namespace MLP_Neural_Network
{
    class Neuron
    {
        public List<double> Wagi { get; set; }
        public double Bias { get; set; }
        public double Wyjscie { get; set; }
        public double PochodnaAktywacji { get; set; }

        public List<double> PoprzednieZmianyWag { get; set; }
        public double PoprzedniaZmianaBiasu { get; set; }

        public Neuron(int liczbaWejsc, bool useBias)
        {
            Wagi = new List<double>();
            PoprzednieZmianyWag = new List<double>(); 
            Random random = new Random();
            for (int i = 0; i < liczbaWejsc; i++)
            {
                Wagi.Add(random.NextDouble() * 2 - 1); //wartosc z zakresu -1 : 1
                PoprzednieZmianyWag.Add(0.0); 
            }
            if (useBias)
            {
                Bias = random.NextDouble() * 2 - 1;
                PoprzedniaZmianaBiasu = 0.0; 
            }
            else
            {
                Bias = 0;
                PoprzedniaZmianaBiasu = 0.0;
            }
            Wyjscie = 0;
            PochodnaAktywacji = 0;
        }

        public static double FunkcjaAktywacji(double x)
        {
            return 1 / (1 + Math.Exp(-x)); //beta = 1  unipolarna
        }

        public static double PochodnaFunkcjiAktywacji(double x)
        {
            double sigmoid = FunkcjaAktywacji(x);
            return sigmoid * (1 - sigmoid);
        }

        public double ObliczWyjscie(List<double> wejscia)
        {
            double suma = 0;
            for (int i = 0; i < wejscia.Count; i++)
            {
                suma += wejscia[i] * Wagi[i];
            }

            if (Bias != 0)
            {
                suma += Bias;
            }

            Wyjscie = FunkcjaAktywacji(suma);
            PochodnaAktywacji = PochodnaFunkcjiAktywacji(suma);
            return Wyjscie;
        }
    }
}