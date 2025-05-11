using System.Globalization;

namespace MLP_Neural_Network
{
    class DataLoader
    {
        public List<List<double>> daneWejscioweNauka = new();
        public List<List<double>> oczekiwaneWyjsciaNauka = new();
        public List<List<double>> daneWejscioweTest = new();
        public List<List<double>> oczekiwaneWyjsciaTest = new();
        public Dictionary<string, List<double>> etykietyKlucze = new();
        public int indeksEtykiety = 0;
        public Dictionary<string, List<List<double>>> danePosortowane = new();
        public readonly static string filePath = "dane.txt";

        public void ReadDataFromFile()
        {
            try
            {
                List<string> linie = File.ReadAllLines(filePath).ToList();

                foreach (var linia in linie)
                {
                    string[] pola = linia.Split(',');
                    if (pola.Length == 5)
                    {
                        List<double> cechy = pola.Take(4)
                            .Select(s => double.Parse(s.Replace(',', '.'), CultureInfo.InvariantCulture))
                            .ToList();
                        string etykieta = pola[4];

                        if (!etykietyKlucze.ContainsKey(etykieta))
                        {
                            etykietyKlucze[etykieta] = new List<double> { 0, 0, 0 };
                            etykietyKlucze[etykieta][indeksEtykiety++] = 1;
                            danePosortowane[etykieta] = new List<List<double>>();
                        }
                        danePosortowane[etykieta].Add(cechy);
                    }
                }

                foreach (var gatunekData in danePosortowane)
                {
                    string gatunek = gatunekData.Key;
                    List<List<double>> daneGatunku = gatunekData.Value;
                    List<double> kluczEtykiety = etykietyKlucze[gatunek];

                    for (int i = 0; i < daneGatunku.Count; i++)
                    {
                        if (i < 40)
                        {
                            daneWejscioweNauka.Add(daneGatunku[i]);
                            oczekiwaneWyjsciaNauka.Add(kluczEtykiety.ToList());
                        }
                        else
                        {
                            daneWejscioweTest.Add(daneGatunku[i]);
                            oczekiwaneWyjsciaTest.Add(kluczEtykiety.ToList());
                        }
                    }
                }

            }
            catch
            {

            }
        }
        public string zmienNaIndex(int indeks)
        {
            switch (indeks)
            {
                case 0: return "0";
                case 1: return "1";
                case 2: return "2";
                default: return "Nieznana";
            }
        }
    }
}
    

