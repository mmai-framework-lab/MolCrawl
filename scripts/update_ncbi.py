import requests
import json
import time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

class NCBIBacteriaUpdaterWebAPI:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.output_dir = Path("assets/genome_species_list/species")
        self.email = "kazunobu.matsubara@riken.jp"  # NCBI推奨：メールアドレス設定

    def search_bacteria_genomes(self, retmax=5000):
        """NCBI E-utilsを使って細菌ゲノムIDを検索"""
        print("🔍 Searching bacteria genomes via NCBI E-utils...")
        
        # 検索クエリ：細菌の完全ゲノムまたは染色体レベル
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'genome',
            'term': 'bacteria[organism] AND ("complete genome"[Properties] OR "chromosome"[Properties])',
            'retmax': retmax,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                genome_ids = data['esearchresult']['idlist']
                print(f"✅ Found {len(genome_ids)} genome records")
                return genome_ids
            else:
                print("❌ No genome IDs found")
                return []
                
        except Exception as e:
            print(f"❌ Error searching genomes: {e}")
            return []
    
    def fetch_genome_details(self, genome_ids, batch_size=100):
        """ゲノムIDから詳細情報を取得"""
        print(f"📋 Fetching details for {len(genome_ids)} genomes...")
        
        all_organisms = set()
        
        # バッチ処理で実行
        for i in range(0, len(genome_ids), batch_size):
            batch_ids = genome_ids[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(genome_ids)-1)//batch_size + 1}")
            
            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                'db': 'genome',
                'id': ','.join(batch_ids),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()
                
                # XMLを解析
                organisms = self.parse_genome_xml(response.text)
                all_organisms.update(organisms)
                
                # NCBI API制限に配慮して少し待機
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ⚠️ Error in batch {i//batch_size + 1}: {e}")
                continue
        
        print(f"✅ Extracted {len(all_organisms)} unique organisms")
        return sorted(list(all_organisms))
    
    def parse_genome_xml(self, xml_content):
        """XMLから生物名を抽出"""
        organisms = set()
        
        try:
            root = ET.fromstring(xml_content)
            
            # 生物名を探す（複数の可能なパス）
            for genome in root.findall('.//Genome'):
                # OrganismName要素を探す
                org_name = genome.find('.//OrganismName')
                if org_name is not None and org_name.text:
                    cleaned = self.clean_organism_name(org_name.text)
                    if cleaned:
                        organisms.add(cleaned)
                
                # 代替パス：ProjectName等
                proj_name = genome.find('.//ProjectName')
                if proj_name is not None and proj_name.text:
                    cleaned = self.clean_organism_name(proj_name.text)
                    if cleaned:
                        organisms.add(cleaned)
                        
        except ET.ParseError as e:
            print(f"  ⚠️ XML parsing error: {e}")
            
        return organisms
    
    def clean_organism_name(self, name):
        """生物名をクリーニングして属レベルに標準化"""
        if not name:
            return None
            
        name = name.strip()
        
        # 不要な情報を除去
        for pattern in [' strain ', ' substrain ', ' subsp.', ' var.', ' serovar ', ' biovar ']:
            if pattern in name:
                name = name.split(pattern)[0]
        
        # 括弧内を除去
        if '(' in name:
            name = name.split('(')[0].strip()
        
        # 数字やコード的な部分を除去
        parts = []
        for part in name.split():
            # 明らかに属名でない部分をスキップ
            if len(part) < 3:
                continue
            if part.lower() in ['sp.', 'bacterium', 'strain', 'isolate']:
                continue
            if part.replace('-', '').replace('_', '').replace('.', '').isdigit():
                continue
            parts.append(part)
        
        if not parts:
            return None
        
        # Candidatus の特別処理
        if len(parts) >= 2 and parts[0].lower() == 'candidatus':
            return f"Candidatus {parts[1]}"
        
        # 通常は属名のみ返す
        return parts[0]
    
    def get_bacteria_via_taxonomy(self):
        """NCBI Taxonomyから細菌の分類を取得"""
        print("🌳 Fetching bacteria taxonomy...")
        
        # 細菌のtaxon ID (2) から下位分類を取得
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'taxonomy',
            'term': 'bacteria[subtree] AND genus[rank]',
            'retmax': 10000,
            'retmode': 'json',
            'email': self.email
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                tax_ids = data['esearchresult']['idlist']
                print(f"✅ Found {len(tax_ids)} bacterial genera")
                
                # 分類名を取得
                genera = self.fetch_taxonomy_names(tax_ids)
                return genera
            else:
                return []
                
        except Exception as e:
            print(f"❌ Error fetching taxonomy: {e}")
            return []
    
    def fetch_taxonomy_names(self, tax_ids, batch_size=200):
        """分類IDから分類名を取得"""
        print("📋 Fetching taxonomy names...")
        
        genera = set()
        
        for i in range(0, len(tax_ids), batch_size):
            batch_ids = tax_ids[i:i+batch_size]
            
            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                'db': 'taxonomy',
                'id': ','.join(batch_ids),
                'retmode': 'xml',
                'email': self.email
            }
            
            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()
                
                # XMLから属名を抽出
                names = self.parse_taxonomy_xml(response.text)
                genera.update(names)
                
                time.sleep(0.3)  # API制限対応
                
            except Exception as e:
                print(f"  ⚠️ Error in taxonomy batch: {e}")
                continue
        
        return sorted(list(genera))
    
    def parse_taxonomy_xml(self, xml_content):
        """Taxonomy XMLから属名を抽出"""
        genera = set()
        
        try:
            root = ET.fromstring(xml_content)
            
            for taxon in root.findall('.//Taxon'):
                # ScientificNameを取得
                sci_name = taxon.find('.//ScientificName')
                if sci_name is not None and sci_name.text:
                    name = sci_name.text.strip()
                    # 属レベルの名前のみ（スペースが含まれていない）
                    if ' ' not in name and len(name) > 2:
                        genera.add(name)
                        
        except ET.ParseError:
            pass
            
        return genera
    
    def update_bacteria_list(self):
        """メイン更新処理"""
        print("🧬 NCBI Bacteria List Updater (Web API)")
        print("=" * 50)
        
        # 方法1: ゲノムデータベースから取得
        genome_ids = self.search_bacteria_genomes(retmax=3000)  # 最新3000件
        organisms_from_genome = []
        if genome_ids:
            organisms_from_genome = self.fetch_genome_details(genome_ids)
        
        # 方法2: 分類データベースから取得
        organisms_from_taxonomy = self.get_bacteria_via_taxonomy()
        
        # 両方の結果をマージ
        all_organisms = set(organisms_from_genome + organisms_from_taxonomy)
        final_list = sorted(list(all_organisms))
        
        print(f"\n📊 Results:")
        print(f"  From genomes: {len(organisms_from_genome)}")
        print(f"  From taxonomy: {len(organisms_from_taxonomy)}")
        print(f"  Combined unique: {len(final_list)}")
        
        # 既存リストとの比較と保存
        self.save_updated_list(final_list)
        
        return True
    
    def save_updated_list(self, new_organisms):
        """更新されたリストを保存"""
        output_file = self.output_dir / "bacteria.txt"
        
        # 既存リストとの比較
        current_organisms = set()
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('//'):
                        current_organisms.add(line)
        
        new_set = set(new_organisms)
        added = new_set - current_organisms
        removed = current_organisms - new_set
        
        print(f"\n📈 Changes:")
        print(f"  Current: {len(current_organisms)}")
        print(f"  New: {len(new_set)}")
        print(f"  Added: {len(added)}")
        print(f"  Removed: {len(removed)}")
        
        # バックアップ
        if output_file.exists():
            backup_file = output_file.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            backup_file.write_text(output_file.read_text())
            print(f"💾 Backup: {backup_file}")
        
        # 新しいリストを保存
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(f"// filepath: {output_file}\n")
            f.write(f"// Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"// Total genera: {len(new_organisms)}\n")
            for organism in new_organisms:
                f.write(f"{organism}\n")
        
        print(f"✅ Updated list saved: {output_file}")
        
        # 変更があった場合は詳細レポート
        if added or removed:
            self.save_change_report(added, removed)

def main():
    updater = NCBIBacteriaUpdaterWebAPI()
    success = updater.update_bacteria_list()
    
    if success:
        print("\n🎉 Update completed successfully!")
    else:
        print("\n❌ Update failed")

if __name__ == "__main__":
    main()