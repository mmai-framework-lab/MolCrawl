#!/usr/bin/env python3
"""
NCBI Taxonomy Complete Updater
全生物群の最新分類リストを取得・更新

対応分類群:
- Bacteria (細菌)
- Archaea (古細菌)
- Fungi (真菌)
- Viridiplantae (植物)
- Metazoa (動物)
  - Vertebrata (脊椎動物)
    - Mammalia (哺乳類)
    - Other Vertebrates (他の脊椎動物)
  - Invertebrates (無脊椎動物)
- Protists (原生生物) - 旧 protozoa
- Viruses (ウイルス)
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET


class NCBITaxonomyCompleteUpdater:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.output_dir = Path("assets/genome_species_list/species")
        self.backup_dir = Path("assets/genome_species_list/backups")
        self.email = "research@riken.jp"  # RIKEN用メールアドレス

        # 現代的なNCBI分類マッピング
        self.taxonomy_groups = {
            # 主要ドメイン
            "bacteria": {
                "taxon_id": "2",
                "search_term": "bacteria[organism]",
                "filename": "bacteria.txt",
                "description": "Bacteria (細菌)",
            },
            "archaea": {
                "taxon_id": "2157",
                "search_term": "archaea[organism]",
                "filename": "archaea.txt",
                "description": "Archaea (古細菌)",
            },
            # 真核生物の主要群
            "fungi": {
                "taxon_id": "4751",
                "search_term": "fungi[organism]",
                "filename": "fungi.txt",
                "description": "Fungi (真菌)",
            },
            "viridiplantae": {
                "taxon_id": "33090",
                "search_term": "viridiplantae[organism]",
                "filename": "plants.txt",
                "description": "Viridiplantae (植物)",
            },
            # 動物群
            "vertebrata": {
                "taxon_id": "7742",
                "search_term": "vertebrata[organism]",
                "filename": "vertebrates.txt",
                "description": "Vertebrata (脊椎動物)",
            },
            "mammalia": {
                "taxon_id": "40674",
                "search_term": "mammalia[organism]",
                "filename": "vertebrate_mammalian.txt",
                "description": "Mammalia (哺乳類)",
            },
            "vertebrate_other": {
                "taxon_id": "7742",
                "search_term": "vertebrata[organism] NOT mammalia[organism]",
                "filename": "vertebrate_other.txt",
                "description": "Other Vertebrates (哺乳類以外の脊椎動物)",
            },
            "invertebrata": {
                "taxon_id": "6656",  # Arthropoda + others
                "search_term": "metazoa[organism] NOT vertebrata[organism]",
                "filename": "invertebrate.txt",
                "description": "Invertebrates (無脊椎動物)",
            },
            # 原生生物 (旧protozoa)
            "protists": {
                "taxon_id": "2759",  # Eukaryota
                "search_term": "eukaryota[organism] NOT metazoa[organism] NOT fungi[organism] NOT viridiplantae[organism]",
                "filename": "protists.txt",
                "description": "Protists (原生生物)",
            },
            # ウイルス
            "viruses": {
                "taxon_id": "10239",
                "search_term": "viruses[organism]",
                "filename": "viruses.txt",
                "description": "Viruses (ウイルス)",
            },
        }

        # 後方互換性のためのマッピング
        self.legacy_mapping = {
            "protozoa.txt": "protists.txt",
        }

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def check_ncbi_taxonomy_updates(self):
        """NCBI分類体系の最新情報をチェック"""
        print("🔍 Checking NCBI Taxonomy structure...")

        # メジャー分類群の統計を取得
        stats = {}

        for group_name, group_info in self.taxonomy_groups.items():
            print(f"  Processing {group_name}...")
            try:
                search_url = f"{self.base_url}/esearch.fcgi"
                params = {
                    "db": "taxonomy",
                    "term": f"{group_info['search_term']} AND genus[rank]",
                    "retmode": "json",  # JSONモードに変更
                    "email": self.email,
                }

                print(f"    URL: {search_url}")
                print(f"    Query: {params['term']}")

                response = requests.get(search_url, params=params)
                response.raise_for_status()

                print(f"    Response status: {response.status_code}")

                # JSONレスポンスを解析
                try:
                    data = response.json()
                    print(f"    Response data: {data}")

                    if "esearchresult" in data and "count" in data["esearchresult"]:
                        count = int(data["esearchresult"]["count"])
                        stats[group_name] = count
                        print(f"    Found {count} genera")
                    else:
                        print("    No count found in response")
                        stats[group_name] = 0

                except json.JSONDecodeError as je:
                    print(f"    JSON decode error: {je}")
                    print(f"    Raw response: {response.text[:200]}...")
                    stats[group_name] = 0

                time.sleep(0.5)  # API制限対応

            except Exception as e:
                print(f"  ⚠️ Error checking {group_name}: {e}")
                stats[group_name] = 0

        print("\n📊 Current NCBI Taxonomy Statistics (Genus level):")
        for group_name, count in stats.items():
            description = self.taxonomy_groups[group_name]["description"]
            print(f"  {description:30}: {count:6,} genera")

        return stats

    def search_taxonomy_by_group(self, group_name, retmax=5000):
        """指定された分類群の属レベル分類を取得"""
        group_info = self.taxonomy_groups[group_name]
        print(f"🔍 Searching {group_info['description']}...")

        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "taxonomy",
            "term": f"{group_info['search_term']} AND genus[rank]",
            "retmax": retmax,
            "retmode": "json",
            "email": self.email,
        }

        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            if "esearchresult" in data and "idlist" in data["esearchresult"]:
                tax_ids = data["esearchresult"]["idlist"]
                count = data["esearchresult"].get("count", len(tax_ids))
                print(f"  ✅ Found {len(tax_ids)}/{count} genera")
                return tax_ids
            else:
                print(f"  ❌ No taxa found for {group_name}")
                return []

        except Exception as e:
            print(f"  ❌ Error searching {group_name}: {e}")
            return []

    def fetch_taxonomy_names_batch(self, tax_ids, batch_size=200):
        """分類IDから分類名をバッチ取得"""
        if not tax_ids:
            return []

        print(f"  📋 Fetching names for {len(tax_ids)} taxa...")

        genera = set()

        for i in range(0, len(tax_ids), batch_size):
            batch_ids = tax_ids[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(tax_ids) - 1) // batch_size + 1

            print(f"    Batch {batch_num}/{total_batches}")

            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                "db": "taxonomy",
                "id": ",".join(batch_ids),
                "retmode": "xml",
                "email": self.email,
            }

            try:
                response = requests.get(fetch_url, params=params)
                response.raise_for_status()

                names = self.parse_taxonomy_xml(response.text)
                genera.update(names)

                time.sleep(0.3)  # NCBI API制限対応

            except Exception as e:
                print(f"    ⚠️ Error in batch {batch_num}: {e}")
                continue

        print(f"  ✅ Extracted {len(genera)} unique genera")
        return sorted(list(genera))

    def parse_taxonomy_xml(self, xml_content):
        """Taxonomy XMLから属名を抽出"""
        genera = set()

        try:
            root = ET.fromstring(xml_content)

            for taxon in root.findall(".//Taxon"):
                # ScientificNameを取得
                sci_name = taxon.find(".//ScientificName")
                if sci_name is not None and sci_name.text:
                    name = sci_name.text.strip()

                    # 属レベルの名前を検証
                    if self.is_valid_genus_name(name):
                        genera.add(name)

        except ET.ParseError as e:
            print(f"    ⚠️ XML parsing error: {e}")

        return genera

    def is_valid_genus_name(self, name):
        """属名として有効かどうかを判定"""
        if not name or len(name) < 2:
            return False

        # スペースが含まれていたら種名レベルなのでスキップ
        if " " in name:
            return False

        # 明らかに属名でない文字列を除外
        invalid_patterns = [
            "sp.",
            "strain",
            "isolate",
            "clone",
            "uncultured",
            "unidentified",
            "environmental",
            "metagenome",
        ]

        name_lower = name.lower()
        for pattern in invalid_patterns:
            if pattern in name_lower:
                return False

        # 数字のみ、または記号のみは除外
        if name.replace("-", "").replace("_", "").isdigit():
            return False

        return True

    def backup_existing_files(self):
        """既存ファイルのバックアップ"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backed_up_files = []

        for group_name, group_info in self.taxonomy_groups.items():
            filename = group_info["filename"]
            file_path = self.output_dir / filename

            if file_path.exists():
                backup_path = self.backup_dir / f"{filename}.backup_{timestamp}"
                backup_path.write_text(file_path.read_text())
                backed_up_files.append(str(backup_path))

        # レガシーファイルもバックアップ
        for legacy_file in self.legacy_mapping.keys():
            legacy_path = self.output_dir / legacy_file
            if legacy_path.exists():
                backup_path = self.backup_dir / f"{legacy_file}.backup_{timestamp}"
                backup_path.write_text(legacy_path.read_text())
                backed_up_files.append(str(backup_path))

        if backed_up_files:
            print(f"💾 Backed up {len(backed_up_files)} files to {self.backup_dir}")

        return backed_up_files

    def compare_with_existing(self, new_genera, filename):
        """既存リストとの比較"""
        file_path = self.output_dir / filename

        current_genera = set()
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("//"):
                        current_genera.add(line)

        new_set = set(new_genera)
        added = new_set - current_genera
        removed = current_genera - new_set
        kept = new_set & current_genera

        return {
            "current_count": len(current_genera),
            "new_count": len(new_set),
            "added": sorted(list(added)),
            "removed": sorted(list(removed)),
            "kept": len(kept),
        }

    def save_taxonomy_file(self, genera_list, group_name):
        """分類リストをファイルに保存"""
        group_info = self.taxonomy_groups[group_name]
        filename = group_info["filename"]
        file_path = self.output_dir / filename

        # 既存リストとの比較
        comparison = self.compare_with_existing(genera_list, filename)

        # ファイル保存
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"// filepath: {file_path}\n")
            f.write(f"// Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"// Group: {group_info['description']}\n")
            f.write(f"// NCBI Taxon ID: {group_info['taxon_id']}\n")
            f.write(f"// Total genera: {len(genera_list)}\n")
            f.write(
                f"// Changes: +{len(comparison['added'])} -{len(comparison['removed'])}\n"
            )
            f.write("\n")

            for genus in genera_list:
                f.write(f"{genus}\n")

        return comparison

    def generate_summary_report(self, all_results):
        """全体の更新結果レポートを生成"""
        report_path = self.output_dir.parent / "taxonomy_update_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# NCBI Taxonomy Update Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            f.write("## Summary\n\n")
            f.write("| Group | Description | Genera | Added | Removed |\n")
            f.write("|-------|-------------|--------|-------|----------|\n")

            total_genera = 0
            total_added = 0
            total_removed = 0

            for group_name, (genera_list, comparison) in all_results.items():
                desc = self.taxonomy_groups[group_name]["description"]
                count = len(genera_list)
                added = len(comparison["added"])
                removed = len(comparison["removed"])

                f.write(
                    f"| {group_name} | {desc} | {count:,} | +{added} | -{removed} |\n"
                )

                total_genera += count
                total_added += added
                total_removed += removed

            f.write(
                f"| **Total** | | **{total_genera:,}** | **+{total_added}** | **-{total_removed}** |\n\n"
            )

            # 詳細変更
            f.write("## Detailed Changes\n\n")
            for group_name, (genera_list, comparison) in all_results.items():
                if comparison["added"] or comparison["removed"]:
                    desc = self.taxonomy_groups[group_name]["description"]
                    f.write(f"### {desc}\n\n")

                    if comparison["added"]:
                        f.write(f"**Added ({len(comparison['added'])}):**\n")
                        for genus in comparison["added"][:20]:  # 最初の20個
                            f.write(f"- {genus}\n")
                        if len(comparison["added"]) > 20:
                            f.write(f"- ... and {len(comparison['added']) - 20} more\n")
                        f.write("\n")

                    if comparison["removed"]:
                        f.write(f"**Removed ({len(comparison['removed'])}):**\n")
                        for genus in comparison["removed"][:20]:
                            f.write(f"- {genus}\n")
                        if len(comparison["removed"]) > 20:
                            f.write(
                                f"- ... and {len(comparison['removed']) - 20} more\n"
                            )
                        f.write("\n")

        print(f"📊 Summary report saved: {report_path}")

    def migrate_legacy_files(self):
        """レガシーファイルの移行"""
        print("🔄 Checking legacy file migration...")

        for legacy_file, new_file in self.legacy_mapping.items():
            legacy_path = self.output_dir / legacy_file
            new_path = self.output_dir / new_file

            if legacy_path.exists() and not new_path.exists():
                print(f"  📁 Migrating {legacy_file} → {new_file}")
                # レガシーファイルをバックアップしてから削除
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"{legacy_file}.migrated_{timestamp}"
                backup_path.write_text(legacy_path.read_text())
                legacy_path.unlink()

    def update_all_taxonomy(self, groups=None):
        """全分類群の更新"""
        print("🧬 NCBI Complete Taxonomy Updater")
        print("=" * 50)

        # 現在の分類統計をチェック
        self.check_ncbi_taxonomy_updates()

        # バックアップ作成
        backed_up = self.backup_existing_files()

        # レガシーファイル移行
        self.migrate_legacy_files()

        # 更新する分類群を決定
        if groups is None:
            groups = list(self.taxonomy_groups.keys())

        all_results = {}

        print(f"\n🔄 Updating {len(groups)} taxonomy groups...")

        for i, group_name in enumerate(groups, 1):
            print(f"\n[{i}/{len(groups)}] Processing {group_name}...")

            # 分類ID取得
            tax_ids = self.search_taxonomy_by_group(group_name)

            if not tax_ids:
                print(f"  ⚠️ Skipping {group_name} - no data found")
                continue

            # 分類名取得
            genera_list = self.fetch_taxonomy_names_batch(tax_ids)

            if not genera_list:
                print(f"  ⚠️ Skipping {group_name} - no genera extracted")
                continue

            # ファイル保存と比較
            comparison = self.save_taxonomy_file(genera_list, group_name)
            all_results[group_name] = (genera_list, comparison)

            # 進捗表示
            print(
                f"  ✅ {group_name}: {len(genera_list)} genera (+{len(comparison['added'])} -{len(comparison['removed'])})"
            )

        # 全体レポート生成
        if all_results:
            self.generate_summary_report(all_results)

        print("\n🎉 Update completed!")
        print(f"📁 Files saved to: {self.output_dir}")
        if backed_up:
            print(f"💾 Backups saved to: {self.backup_dir}")

        return all_results

    def test_simple_query(self):
        """シンプルなクエリでNCBI接続をテスト"""
        print("🔬 Testing simple NCBI connection...")

        test_url = f"{self.base_url}/esearch.fcgi"
        params = {
            "db": "taxonomy",
            "term": "bacteria[organism]",
            "retmax": 5,
            "retmode": "json",
            "email": self.email,
        }

        try:
            print(f"  URL: {test_url}")
            print(f"  Params: {params}")

            response = requests.get(test_url, params=params, timeout=30)
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"  Response: {json.dumps(data, indent=2)}")
                    return True
                except (ValueError, KeyError):
                    print(f"  Raw text: {response.text}")
                    return False
            else:
                print(f"  Error: {response.text}")
                return False

        except Exception as e:
            print(f"  Exception: {e}")
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Update NCBI taxonomy lists")
    parser.add_argument(
        "--groups",
        nargs="*",
        choices=list(NCBITaxonomyCompleteUpdater().taxonomy_groups.keys()),
        help="Specific groups to update (default: all)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check current statistics without updating",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test NCBI connection with simple query",
    )

    args = parser.parse_args()

    updater = NCBITaxonomyCompleteUpdater()

    if args.test_connection:
        success = updater.test_simple_query()
        if not success:
            print("❌ Connection test failed")
        else:
            print("✅ Connection test successful")
    elif args.check_only:
        updater.check_ncbi_taxonomy_updates()
    else:
        updater.update_all_taxonomy(groups=args.groups)


if __name__ == "__main__":
    main()
