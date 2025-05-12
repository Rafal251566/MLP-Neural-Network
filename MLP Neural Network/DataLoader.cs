using System.Globalization;
using System.Text.RegularExpressions;

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
        public Dictionary<string, List<List<double>>> danePosortowaneWejscia = new(); 
        public Dictionary<string, List<List<double>>> danePosortowaneWyjscia = new(); 
        public readonly static string filePath = "wzorce.txt";

        public void ReadDataFromFile()
        {
            try
            {
                List<string> linie = File.ReadAllLines(filePath).ToList();

                foreach (var linia in linie)
                {
                    Match match = Regex.Match(linia, @"\(\((.*?)\),\((.*?)\)\)");
                    if (match.Success)
                    {
                        string wejsciaString = match.Groups[1].Value;
                        string wyjsciaString = match.Groups[2].Value;

                        List<double> wejscia = wejsciaString.Split(',')
                            .Select(s => double.Parse(s.Replace(',', '.'), CultureInfo.InvariantCulture))
                            .ToList();

                        List<double> wyjscia = wyjsciaString.Split(',')
                            .Select(s => double.Parse(s.Replace(',', '.'), CultureInfo.InvariantCulture))
                            .ToList();

                        string kluczWyjscia = string.Join(",", wyjscia);

                        if (!danePosortowaneWejscia.ContainsKey(kluczWyjscia))
                        {
                            danePosortowaneWejscia[kluczWyjscia] = new List<List<double>>();
                            danePosortowaneWyjscia[kluczWyjscia] = new List<List<double>>();
                            etykietyKlucze[kluczWyjscia] = new List<double>(wyjscia);
                        }
                        danePosortowaneWejscia[kluczWyjscia].Add(wejscia);
                        danePosortowaneWyjscia[kluczWyjscia].Add(wyjscia);
                    }
                }

                foreach (var klasaData in danePosortowaneWejscia)
                {
                    string kluczKlasy = klasaData.Key;
                    List<List<double>> wejsciaKlasy = klasaData.Value;
                    List<List<double>> wyjsciaKlasy = danePosortowaneWyjscia[kluczKlasy];
                    List<double> oczekiwaneWyjscieKlasy = etykietyKlucze[kluczKlasy];

                    for (int i = 0; i < wejsciaKlasy.Count; i++)
                    {
                        if (i < 40)
                        {
                            daneWejscioweNauka.Add(wejsciaKlasy[i]);
                            oczekiwaneWyjsciaNauka.Add(oczekiwaneWyjscieKlasy.ToList());
                        }
                        else
                        {
                            daneWejscioweTest.Add(wejsciaKlasy[i]);
                            oczekiwaneWyjsciaTest.Add(oczekiwaneWyjscieKlasy.ToList());
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Wystąpił błąd podczas odczytu pliku: {ex.Message}");
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