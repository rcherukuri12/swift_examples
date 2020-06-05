//A reader for CSV with header,string labels and numeric dataset. 
// TO DO : Make it automatic to detect for string or numeric labels 
import Foundation
struct csv_reader{
    var filePath  :String
    var labels:[String]?
    ///////////////////////////////////////////////////////
    // This method reads the csv data.
    // return values :
    // X -  array of floats
    // yi - index of labels in Int
    mutating func load_data(label:String)-> (X:[[Float]],yi:[Int]) {
    var Arr:[String] = []
    var X:[[Float]]=[[]]
    var y:[String] = []
    do{       
        let url  = URL(fileURLWithPath:filePath)
        let data = try Data(contentsOf:url)
        let dataencoded = String(data:data,encoding:.utf8)
        let dataArr = dataencoded?.components(separatedBy:"\n")
        guard let da = dataArr else { return ([[]],[]) }
        for line in da{
            let trimmedString = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedString.isEmpty {
                Arr.append(trimmedString)
            }
        }
    }catch{
        print("possible error:\(error)")
    }
    let headers = Arr[0].split(separator: ",")
    guard let label_i = headers.firstIndex(where: { $0.hasPrefix(label) }) else { return ([[]],[]) }
    // now that we got the headers, remove it.
    Arr.remove(at:0)
    // for the rest of the array split into X and y   
    let label_index = Int(label_i)  
    for row in Arr {
       var items = row.split(separator: ",")
       if items.count > label_index {
            let s = String(items[label_index])
            y.append(s)
            // now remove that label
            items.remove(at:label_index)
            // take the remaining data as float features
            X.append(items.map {Float($0)!})
        }
    }
    X.remove(at:0)
    return (X,convertLabels2Indices(y:y))
    }

    private mutating func convertLabels2Indices(y:[String])->[Int]{
        let label_names = Array(Set<String>(y))
        self.labels = label_names
        return y.map{label_names.firstIndex(of:$0)!}
    }

    func getLabels()->[String]{
        guard let unwrapped_labels = self.labels else{
            return [""]
        }
        return unwrapped_labels
    }
}